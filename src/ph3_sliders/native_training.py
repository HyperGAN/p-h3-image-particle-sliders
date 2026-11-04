"""Full-state H3 host binding to the unchanged classic Anima Game."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import tempfile

import torch
from particle_sliders_anima_gmix import Game, TRAINING_CONFIG, fit_normalization, init_ema, update_ema

from .evaluation import aggregate_rows, response_statistics
from .learning_rates import LearningRates, assert_learning_rates, bind_learning_rates
from .native_cache import file_hash, tensor_hash


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return deepcopy(value)


class NativeTrainingRun:
    def __init__(self, host, training_cache, *, rates=LearningRates(), seed=17, identity):
        if host.adapter is None or training_cache.identity["split"] != "train":
            raise ValueError("owned attached adapter and training-only cache required")
        if training_cache.positions != len(host.timesteps) or any(
                row["time"] != float(host.timesteps[row["position"]]) for row in training_cache.records):
            raise ValueError("cache and native host schedule differ")
        self.host, self.cache, self.rates = host, training_cache, rates
        self.adapter = host.adapter
        normalization = fit_normalization(training_cache, positions=training_cache.positions)
        self.game = Game(self.adapter, normalization, len(training_cache), seed=seed, microbatch=1)
        bind_learning_rates(self.game, rates)
        self.ema = init_ema(self.adapter)
        self.completed_steps, self.last_metrics = 0, None
        self.failed_step = False
        self.positions = [r["position"] for r in self.cache.records]
        self.signature = dict(schema="ph3-native-training-cohort-v1", identity=deepcopy(identity),
            cache=training_cache.index["fingerprint"], rates=asdict(rates), seed=seed,
            normalization=tensor_hash({k: normalization[k] for k in ("scale", "edit_rms")}),
            targets=list(self.adapter.names), response_surface="full_packed_image_then_audio_velocity",
            recipe=dict(TRAINING_CONFIG, g_lr=rates.generator, d_lr=rates.critic, particle_lr=rates.particles),
            sigmas=list(host.sigmas), latent_shape=[host.latent_height, host.latent_width])

    def residual(self, indices):
        responses = []
        for index in indices:
            row = self.cache[index]
            velocity = self.host.velocity(row["state"], self.cache.contexts[row["neutral_context"]], row["position"])
            responses.append((self.host.response(velocity) - row["positive"].to(self.host.device)).flatten())
        return torch.stack(responses)

    def step(self):
        if self.failed_step:
            raise RuntimeError("restore a complete checkpoint after an interrupted Game update")
        assert_learning_rates(self.game, self.rates)
        try:
            with self.adapter.scaled(1.):
                metrics = self.game.update(self.residual, self.positions, self.completed_steps + 1)
        except BaseException:
            self.failed_step = True
            raise
        update_ema(self.ema, self.adapter)
        self.completed_steps += 1
        self.last_metrics = cpu_state(metrics)
        return self.last_metrics

    def snapshot(self):
        if self.failed_step:
            raise RuntimeError("partial Game update is not a complete checkpoint")
        assert_learning_rates(self.game, self.rates)
        return cpu_state(dict(schema="ph3-native-full-training-state-v1", signature=self.signature,
            adapter=self.adapter.state_dict(), game=self.game.state_dict(), ema=self.ema,
            completed_steps=self.completed_steps, last_metrics=self.last_metrics,
            rng_cpu=torch.get_rng_state(), rng_cuda=(torch.cuda.get_rng_state(self.host.device)
                if self.host.device.type == "cuda" else None)))

    def restore(self, packet):
        if packet.get("schema") != "ph3-native-full-training-state-v1" or packet.get("signature") != self.signature:
            raise ValueError("native source/model/cache/rate cohort differs")
        if type(packet.get("completed_steps")) is not int or packet["completed_steps"] < 0:
            raise ValueError("invalid completed native update count")
        # Validate all adapter/EMA leaves before mutating the live run.
        expected = self.adapter.state_dict()
        for key in ("adapter", "ema"):
            if set(packet[key]) != set(expected):
                raise ValueError("native adapter keys differ")
            for name, value in packet[key].items():
                if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not torch.isfinite(value).all():
                    raise ValueError("invalid native adapter tensor")
        groups = packet["game"]["g"]["param_groups"]
        dgroups = packet["game"]["d"]["param_groups"]
        if [g["lr"] for g in groups] != [self.rates.generator, self.rates.particles] or [g["lr"] for g in dgroups] != [self.rates.critic]:
            raise ValueError("checkpoint optimizer rates differ")
        self.adapter.load_state_dict(packet["adapter"], strict=True)
        self.game.load_state_dict(packet["game"])
        self.ema = {k: v.to(self.host.device).clone() for k, v in packet["ema"].items()}
        self.completed_steps, self.last_metrics = packet["completed_steps"], deepcopy(packet["last_metrics"])
        self.failed_step = False
        torch.set_rng_state(packet["rng_cpu"])
        if self.host.device.type == "cuda":
            torch.cuda.set_rng_state(packet["rng_cuda"], self.host.device)
        assert_learning_rates(self.game, self.rates)

    def save(self, directory):
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
        path = temporary / "training.pt"
        torch.save(self.snapshot(), path)
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())
        receipt = dict(schema="ph3-native-checkpoint-receipt-v1", completed_steps=self.completed_steps,
            signature=self.signature, training_sha256=file_hash(path))
        (temporary / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, directory)
        return receipt

    def load(self, directory):
        directory = Path(directory)
        receipt = json.loads((directory / "manifest.json").read_text())
        if file_hash(directory / "training.pt") != receipt["training_sha256"]:
            raise ValueError("checkpoint file changed")
        self.restore(torch.load(directory / "training.pt", map_location="cpu", weights_only=True))

    def export(self, directory, *, selection="ema"):
        from safetensors.torch import save_file
        if selection not in ("raw", "ema"):
            raise ValueError("select raw or EMA")
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=directory.name + ".partial-", dir=directory.parent))
        values = cpu_state(self.ema if selection == "ema" else self.adapter.state_dict())
        save_file({k: v.contiguous() for k, v in values.items()}, str(temporary / "adapter.safetensors"))
        receipt = dict(schema="ph3-native-routed-adapter-v1", signature=self.signature,
            completed_steps=self.completed_steps, selection=selection,
            weights_sha256=file_hash(temporary / "adapter.safetensors"), rank=8,
            particle_shape=[128, 4], strength_range=[0., 1.], distillation=False)
        (temporary / "manifest.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, directory)
        return receipt

    def evaluate(self, cache, *, selection="ema", raw_path=None):
        if selection not in ("raw", "ema"):
            raise ValueError("select raw or EMA")
        original = cpu_state(self.adapter.state_dict())
        modes = [(module, module.training) for module in self.host.model.modules()]
        devices = [self.host.device.index or 0] if self.host.device.type == "cuda" else []
        rows, raw = [], {}
        try:
            self.host.model.eval()
            if selection == "ema":
                self.adapter.load_state_dict(self.ema, strict=True)
            with torch.random.fork_rng(devices=devices), torch.no_grad(), self.adapter.scaled(1.):
                for row in cache.records:
                    velocity = self.host.velocity(row["state"], cache.contexts[row["neutral_context"]], row["position"])
                    student = self.host.response(velocity)
                    label = {k: row[k] for k in ("scene", "seed", "path", "position")}
                    rows.append(dict(label, **response_statistics(student, row["neutral"], row["positive"])))
                    if raw_path is not None:
                        raw[f"student.{len(rows)-1}"] = student.detach().cpu().contiguous()
            result = dict(aggregate_rows(rows, expected_examples=cache.examples,
                expected_positions=range(cache.positions)), selection=selection,
                completed_steps=self.completed_steps, cache_fingerprint=cache.index["fingerprint"],
                native_h3_qualified=bool(self.signature["identity"].get("native_qualification_sha256")))
            if raw_path is not None:
                from safetensors.torch import save_file
                raw_path = Path(raw_path)
                if raw_path.exists():
                    raise FileExistsError(raw_path)
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(raw, str(raw_path), metadata=dict(cache_fingerprint=cache.index["fingerprint"],
                    selection=selection, completed_steps=str(self.completed_steps),
                    row_labels=json.dumps([{k: r[k] for k in ("scene", "seed", "path", "position")} for r in rows])))
                result["raw_student_responses"] = dict(path=str(raw_path), sha256=file_hash(raw_path))
            return result
        finally:
            self.adapter.load_state_dict(original, strict=True)
            for module, training in modes:
                module.training = training
