"""Actual H3 packed-state host, using the pinned upstream layout and scheduler.

No Hub access or implicit allocation: callers supply an owned frozen model and
already encoded conditioning. Both image and coupled audio velocities are kept.
"""

from dataclasses import dataclass
from contextlib import nullcontext

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.p_h3_image.before_denoise import (
    PH3ImagePrepareLayoutStep, PH3ImageSetTimestepsStep, patchify_video_latents,
)

SIGMAS = (1., .984753, .967905, .951688, .925372, .881923, .810083, .528718, 0.)


@dataclass
class PackedState:
    image: torch.Tensor
    audio: torch.Tensor

    def copy(self, device="cpu"):
        return PackedState(*(x.detach().to(device=device, dtype=torch.float32).clone()
                             for x in (self.image, self.audio)))


@dataclass
class Conditioning:
    name: str
    embeddings: torch.Tensor
    layout: dict

    def copy(self):
        return Conditioning(self.name, self.embeddings.detach().cpu().clone(),
                            {k: v.detach().cpu().clone() for k, v in self.layout.items()})


def attention_targets(model):
    """Explicit native main/refiner names, checked against actual projections."""
    names = []
    for prefix, count in (("transformer_blocks", model.config.num_layers),
                          ("token_refiner.refiner_blocks", model.config.num_refiner_layers)):
        for index in range(count):
            for projection in ("to_q", "to_k", "to_v", "to_out.0"):
                name = f"{prefix}.{index}.attn.{projection}"
                module = model.get_submodule(name)
                width = model.config.num_attention_heads * model.config.attention_head_dim
                expected = ((width, model.config.hidden_size) if projection == "to_out.0"
                            else (model.config.hidden_size, width))
                if not isinstance(module, torch.nn.Linear) or (module.in_features, module.out_features) != expected:
                    raise ValueError(f"native projection geometry differs: {name}")
                names.append(name)
    return tuple(names)


class H3Host:
    def __init__(self, model, *, latent_height, latent_width, device="cpu", sigmas=SIGMAS):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("native base must already be frozen")
        self.model, self.device = model, torch.device(device)
        self.latent_height, self.latent_width = latent_height, latent_width
        self.sigmas = tuple(sigmas)
        self.adapter = None
        self.timesteps = self.scheduler().timesteps.clone().cpu()
        self.targets = attention_targets(model)

    def scheduler(self):
        scheduler = MiniMaxH3Scheduler(shift=12., shift_provided_sigmas=False)
        scheduler.set_timesteps(sigmas=self.sigmas, device=self.device)
        return scheduler

    def conditioning(self, name, embeddings):
        if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[2] != self.model.config.text_dim:
            raise ValueError("native single-example text conditioning geometry differs")
        if not torch.isfinite(embeddings).all():
            raise ValueError("finite text conditioning required")
        values = PH3ImagePrepareLayoutStep.build_packed_sequence(
            text_token_tags=torch.ones(embeddings.shape[1], dtype=torch.long),
            num_latent_frames=1, latent_height=self.latent_height, latent_width=self.latent_width,
            num_audio_latents=2, patch_size=tuple(self.model.config.patch_size),
            audio_channels=2, audio_tag=2, video_tag=0)
        keys = ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices")
        return Conditioning(name, embeddings.detach().cpu().clone(), dict(zip(keys, values[:5])))

    def initial_state(self, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = torch.randn(1, self.model.config.in_channels, 1, self.latent_height, self.latent_width,
                            generator=generator, dtype=torch.float32)
        # Match native request RNG ordering: image noise first, coupled audio next.
        audio = torch.randn(4, self.model.config.audio_in_channels, generator=generator, dtype=torch.float32)
        return PackedState(patchify_video_latents(image, tuple(self.model.config.patch_size)), audio).copy(self.device)

    def velocity(self, state, conditioning, position):
        if type(position) is not int or not 0 <= position < len(self.timesteps):
            raise ValueError("declared native schedule position required")
        layout = {k: v.to(self.device) for k, v in conditioning.layout.items()}
        time = float(self.timesteps[position])
        times, indices = PH3ImageSetTimestepsStep.build_row_timesteps(
            video_indices=layout["video_indices"], audio_indices=layout["audio_indices"],
            num_condition_video_rows=0, num_condition_audio_rows=0,
            num_text_tokens=layout["text_indices"].numel(), video_timestep=time,
            audio_timestep=time, condition_video_timestep=max(time, .999), condition_audio_timestep=1.)
        video, audio = self.model(hidden_states=state.image.to(self.device)[None],
            audio_hidden_states=state.audio.to(self.device)[None],
            encoder_hidden_states=conditioning.embeddings.to(self.device),
            timestep=times.to(self.device), timestep_indices=indices.to(self.device),
            return_dict=False, **layout)
        return PackedState(video[0].float(), audio[0].float())

    @staticmethod
    def response(velocity):
        return torch.cat((velocity.image.flatten(), velocity.audio.flatten()))[None]

    def step(self, state, velocity, position, video_scheduler, audio_scheduler):
        # Use upstream's exact two-stage blend, including timestep/grid rounding.
        return PackedState(video_scheduler.step(velocity.image.float(), video_scheduler.timesteps[position],
                                                state.image, return_dict=False)[0],
                           audio_scheduler.step(velocity.audio.float(), audio_scheduler.timesteps[position],
                                                state.audio, return_dict=False)[0])

    @torch.no_grad()
    def rollout(self, conditioning, seed, *, strength=0., observer=None):
        video_scheduler, audio_scheduler = self.scheduler(), self.scheduler()
        state = self.initial_state(seed)
        with self.adapter.scaled(strength) if self.adapter is not None else nullcontext():
            for position in range(len(self.timesteps)):
                velocity = self.velocity(state, conditioning, position)
                if observer is not None:
                    observer(position, state, velocity)
                state = self.step(state, velocity, position, video_scheduler, audio_scheduler)
        return state

    @torch.no_grad()
    def collect_pair(self, neutral, positive, *, scene, seed, final_states=None):
        """Frozen same-noise neutral/positive paths, both teachers on every state."""
        records = []
        for path, context in (("neutral", neutral), ("positive", positive)):
            def observe(position, state, rollout_velocity):
                n = rollout_velocity if path == "neutral" else self.velocity(state, neutral, position)
                t = rollout_velocity if path == "positive" else self.velocity(state, positive, position)
                records.append(dict(scene=scene, seed=seed, path=path, position=position,
                    time=float(self.timesteps[position]), state=state.copy(),
                    neutral_context=neutral.name, positive_context=positive.name,
                    neutral=self.response(n).cpu().clone(), positive=self.response(t).cpu().clone()))
            final = self.rollout(context, seed, strength=0., observer=observe)
            if final_states is not None:
                final_states[path] = final.copy()
        return records

    def unpack_image(self, state):
        pt, ph, pw = self.model.config.patch_size
        channels = self.model.config.in_channels
        rows = state.image.reshape(1, 1 // pt, self.latent_height // ph,
            self.latent_width // pw, channels, pt, ph, pw)
        return rows.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
            1, channels, 1, self.latent_height, self.latent_width).contiguous()
