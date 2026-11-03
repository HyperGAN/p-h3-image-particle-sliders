"""Unrelated CPU fixture around the extracted released-Anima gmix runner.

Only adapter/cache/host ownership and bounded fixture state live here. Every
D/G/VIC update, optimizer and noise/row stream comes from the shared runner.
Synthetic snapshots are not a native H3 checkpoint or inference format.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
from particle_sliders_anima_gmix import Game, TRAINING_CONFIG, init_ema, update_ema

from .paired_cache import PairedCache
from .synthetic import SyntheticProjection

FIXTURE = "unrelated-classic-gmix-v1"
CORE_REVISION = "a119ca1ecd3d5d6c437065839d22739b04f2f4d8"
RUNNER_REVISION = "426fc9297e407a890c88163b9e21075efa78bde3"


@dataclass(frozen=True)
class Config:
    steps: int = 4
    positions: int = 8
    seed: int = 7

    def __post_init__(self):
        for name, low, high in (("steps", 1, 16), ("positions", 2, 10), ("seed", 0, 2**31 - 1)):
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(name + " exceeds the unrelated CPU fixture budget")


def _tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


class SyntheticGmixRun:
    """No external model, records, files, overrides, references or devices accepted."""

    def __init__(self, config=Config()):
        if type(config) is not Config:
            raise ValueError("fixed synthetic Config required; no staged model or options")
        self.config = config
        self.closed = False
        self.completed_steps = 0
        self.last_metrics = None
        self.fixture = None
        try:
            with torch.device("cpu"), torch.random.fork_rng(devices=[]):
                torch.set_rng_state(torch.Generator(device="cpu").manual_seed(config.seed).get_state())
                self.fixture = SyntheticProjection(config.seed)
                self.adapter = self.fixture.adapter
                generator = torch.Generator(device="cpu").manual_seed(config.seed + 100)
                records = []
                with torch.no_grad(), self.adapter.scaled(0):
                    for example in range(4):
                        for position in range(config.positions):
                            latent = torch.randn(1, 8, 2, 2, generator=generator, device="cpu", dtype=torch.float32)
                            neutral_context = torch.linspace(-.1, .1, 8, device="cpu", dtype=torch.float32)[None] + example * .03
                            edit = torch.linspace(.03, .09, 8, device="cpu", dtype=torch.float32)[None]
                            positive_context = neutral_context + edit * (example + 1) * (1 + position * .1)
                            time = position / (config.positions - 1)
                            records.append(dict(position=position, time=time, latent=latent,
                                conditioning=neutral_context,
                                neutral=self._velocity(latent, neutral_context, time).detach(),
                                positive=self._velocity(latent, positive_context, time).detach()))
                self.cache = PairedCache(records, positions=config.positions,
                    identity=dict(split="train", fixture=FIXTURE, no_native_model=True))
                self.normalization = self.cache.normalization()
                self.positions = [self.cache[i]["position"] for i in range(len(self.cache))]
                self.game = Game(self.adapter, self.normalization, len(self.cache),
                                 seed=config.seed, microbatch=1)
                self.ema = init_ema(self.adapter)
            self.signature = dict(fixture=FIXTURE, config=asdict(config), core_revision=CORE_REVISION,
                runner_revision=RUNNER_REVISION, cache=self.cache.index["fingerprint"],
                normalization=_tensor_hash({k: self.normalization[k] for k in ("scale", "edit_rms")}),
                base=_tensor_hash(self.fixture.host.state_dict()),
                product_sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__), Path(__file__).with_name("adapter.py"),
                    Path(__file__).with_name("paired_cache.py"), Path(__file__).with_name("synthetic.py"))},
                recipe=deepcopy(TRAINING_CONFIG), torch=str(torch.__version__))
        except BaseException:
            self.close()
            raise

    def _velocity(self, latent, conditioning, time):
        return self.fixture(latent.movedim(1, -1)).movedim(-1, 1) + conditioning[:, :, None, None] + time * .01

    def residual(self, indices):
        # Full fixture velocity residual; no pooling or rank8 feature projection.
        result = []
        for index in indices:
            record = self.cache[index]
            student = self._velocity(record["latent"], record["conditioning"], record["time"])
            result.append((student - record["positive"]).flatten())
        return torch.stack(result)

    def step(self):
        self._open()
        if self.completed_steps >= self.config.steps:
            raise RuntimeError("synthetic update budget exhausted")
        try:
            with self.adapter.scaled(1):
                metrics = self.game.update(self.residual, self.positions, self.completed_steps + 1)
            update_ema(self.ema, self.adapter)
            self.completed_steps += 1
            self.last_metrics = deepcopy(metrics)
            return metrics
        except BaseException:
            self.close()
            raise

    def train(self, stop_after=None):
        stop = self.config.steps if stop_after is None else stop_after
        if type(stop) is not int or not self.completed_steps < stop <= self.config.steps:
            raise ValueError("stop must advance within the original fixed fixture budget")
        while self.completed_steps < stop:
            self.step()
        return deepcopy(self.last_metrics)

    def snapshot(self):
        self._open()
        return deepcopy(dict(schema="synthetic-anima-gmix-state-v1", signature=self.signature,
            adapter=self.adapter.state_dict(), game=self.game.state_dict(), ema=self.ema,
            completed_steps=self.completed_steps, metrics=self.last_metrics))

    def restore(self, packet):
        self._open()
        packet = deepcopy(packet)
        if (not isinstance(packet, dict) or set(packet) != {"schema", "signature", "adapter", "game", "ema", "completed_steps", "metrics"}
                or packet["schema"] != "synthetic-anima-gmix-state-v1" or packet["signature"] != self.signature):
            raise ValueError("synthetic source/cohort/state differs; no generation migration")
        step = packet["completed_steps"]
        if type(step) is not int or not 0 <= step <= self.config.steps:
            raise ValueError("invalid synthetic state step")
        if (step == 0 and packet["metrics"] is not None or step > 0
                and (not isinstance(packet["metrics"], dict) or packet["metrics"].get("step") != step)):
            raise ValueError("synthetic metrics/state step differs")
        for key in ("adapter", "ema"):
            expected = self.adapter.state_dict()
            if set(packet[key]) != set(expected):
                raise ValueError("synthetic adapter/EMA keys differ")
            for name, value in packet[key].items():
                if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                        or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
                    raise ValueError("invalid synthetic adapter/EMA tensor")
        # Exercise the original loader on a fresh owned CPU fixture before any
        # live state changes. This is a fixture packet, not native artifact IO.
        with SyntheticGmixRun(self.config) as trial:
            trial.adapter.load_state_dict(packet["adapter"], strict=True)
            trial.game.load_state_dict(packet["game"])
        self.adapter.load_state_dict(packet["adapter"], strict=True)
        self.game.load_state_dict(packet["game"])
        self.ema = deepcopy(packet["ema"])
        self.completed_steps, self.last_metrics = step, deepcopy(packet["metrics"])

    def evaluate(self, *, strength=1., selection="raw", index=0):
        self._open()
        if selection not in ("raw", "ema"):
            raise ValueError("choose raw or EMA synthetic weights")
        original = deepcopy(self.adapter.state_dict())
        try:
            if selection == "ema":
                self.adapter.load_state_dict(self.ema, strict=True)
            record = self.cache[index]
            with torch.no_grad(), self.adapter.scaled(strength):
                return self._velocity(record["latent"], record["conditioning"], record["time"]).detach().clone()
        finally:
            self.adapter.load_state_dict(original, strict=True)

    def export(self, *, selection="ema"):
        self._open()
        if selection not in ("raw", "ema"):
            raise ValueError("choose raw or EMA synthetic weights")
        return deepcopy(dict(schema="synthetic-anima-gmix-adapter-v1", signature=self.signature,
            selection=selection, completed_steps=self.completed_steps,
            adapter=self.ema if selection == "ema" else self.adapter.state_dict(), native_h3=False))

    def _open(self):
        if self.closed:
            raise RuntimeError("synthetic run is closed")

    def close(self):
        if not self.closed:
            if self.fixture is not None:
                self.fixture.close()
            self.closed = True

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()


def check(steps=4):
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with SyntheticGmixRun(Config(steps=steps)) as run:
            metrics = run.train()
            return dict(status="passed", device="cpu", fixture=FIXTURE,
                engine=type(run.game).__module__ + "." + type(run.game).__name__,
                formulation_id="particle-gmix-1600-v2", completed_steps=run.completed_steps,
                response_width=run.game.scale.shape[1], rank=8, particle_shape=list(run.adapter.particles.shape),
                g_lr=run.game.g.param_groups[0]["lr"], d_lr=run.game.d.param_groups[0]["lr"],
                particle_lr=run.game.g.param_groups[1]["lr"], ema_decay=TRAINING_CONFIG["ema"],
                last_metrics=metrics, auxiliary_losses=[], artifacts_written=False,
                model_weights_loaded=False, native_h3_qualified=False,
                published_run_parity=False, noise_hold="pinned source stamp: edit_rms*1.3")
    finally:
        torch.set_num_threads(previous_threads)
