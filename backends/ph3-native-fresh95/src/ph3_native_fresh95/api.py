"""New cached fresh/FULL/RAW route; original coupled native game and objects."""

from copy import deepcopy
from pathlib import Path

import torch
from particlegan import get_recipe
from particle_sliders.pg095 import ConditionalPG095, PG_SOURCE_PROFILE, PG_VERSION

from . import PROFILE, ownership, policy
from ._config import configuration
from .cache import host_identity, load_cache, validate_identity, validate_row
from .util import equal, new_directory, read_packet, save_packet, write_json

FULL = "PH3Image-cached-explicitfresh95-typed-FULL-v1"
RAW = "PH3Image-cached-explicitfresh95-typed-RAW-v1"


class ReceiptBackend:
    def __init__(self, original, receipt, sources, cache_identity):
        self.original, self.network = original, original.network
        self.receipt, self.sources, self.cache_identity = receipt, sources, cache_identity

    def __getattr__(self, name):
        return getattr(self.original, name)

    def metadata(self):
        return {
            **self.original.metadata(),
            "fresh_profile": PROFILE,
            "initialization": deepcopy(self.receipt),
            "fresh_sources": deepcopy(self.sources),
            "completed_cache_identity": deepcopy(self.cache_identity),
            "selection": "raw",
            "ema_decay": None,
        }


def _admit(callback, *, operation, cohort, identity, cache_identity, completed_steps):
    request = {
        "operation": operation, "profile": PROFILE, "cohort_id": cohort,
        "model_identity": deepcopy(identity), "cache_identity": deepcopy(cache_identity),
        "completed_steps": completed_steps, "selection": "raw", "ema_decay": None,
        "requires_authentic_completed_cache_owner_binding": True,
        "stop_on_established_qualified_own_failure_or_owner_STOP": True,
        "unmeasured_is_not_PASS_or_eligibility": True,
    }
    if not callable(callback) or callback(request) is not True:
        raise ValueError("explicit external owner admission/negative-only STOP decision required")


def _receipt(value, *, cohort, config, sources):
    keys = {"schema", "profile", "cohort_id", "config", "sources", "seed", "parameter_policies", "existing_Parameter_objects_preserved", "pretrained_parameters_touched", "critic_policy", "new_optimizer_count_at_receipt", "weight_EMA", "old_cohort_migration"}
    if not isinstance(value, dict) or set(value) != keys or value.get("schema") != "PH3Image-native-explicit-fresh95-before-optimizers-v1" or value.get("profile") != PROFILE or value.get("cohort_id") != cohort or value.get("config") != config or value.get("sources") != sources or value.get("weight_EMA") is not None or value.get("new_optimizer_count_at_receipt") != 0 or value.get("existing_Parameter_objects_preserved") is not True or value["pretrained_parameters_touched"] is not False or value["old_cohort_migration"] is not False or value["seed"] != config["seed"]:
        raise ValueError("exact saved fresh receipt required; never initialize after restore")
    rows = value["parameter_policies"]
    roles = {"original-constructor-KEEP", "protected-zero-UP", "down-sqrt5-constructor-scale-QR", "bank-R2Normal-std1"}
    if not isinstance(rows, list) or not rows or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("unique complete fresh parameter-policy rows required")
    for row in rows:
        if set(row) != {"name", "role", "distribution", "shape", "dtype", "seed", "before_sha256", "after_sha256"} or row["role"] not in roles or row["dtype"] != "torch.float32" or any(type(n) is not int or n < 1 for n in row["shape"]) or any(not isinstance(row[k], str) or len(row[k]) != 64 or any(c not in "0123456789abcdef" for c in row[k]) for k in ("before_sha256", "after_sha256")):
            raise ValueError("exact typed initialization policy required")
        if row["role"] in ("original-constructor-KEEP", "protected-zero-UP") and row["before_sha256"] != row["after_sha256"]:
            raise ValueError("saved constructor KEEP/zero-UP evidence differs")
    banks = [r for r in rows if r["role"] == "bank-R2Normal-std1"]
    if len(banks) != 1 or banks[0]["shape"] != [128, 4]:
        raise ValueError("one original128x4/std1 bank policy required")


def _metadata(packet, schema, *, identity, host, sources):
    common = {"schema", "profile", "cohort_id", "config", "sources", "identity", "host", "cache_identity", "initialization", "selection", "ema_decay", "completed_steps"}
    expected = common | ({"game"} if schema == FULL else {"network", "backend_metadata"})
    if not isinstance(packet, dict) or set(packet) != expected or packet["schema"] != schema or packet["profile"] != PROFILE or packet["identity"] != identity or packet["host"] != host_identity(host) or packet["sources"] != sources or packet["selection"] != "raw" or packet["ema_decay"] is not None:
        raise ValueError("strict new-profile whole packet/model/native/Source identity required")
    config = packet["config"]
    if configuration(**config) != config or type(packet["completed_steps"]) is not int or not 0 <= packet["completed_steps"] <= config["total_steps"]:
        raise ValueError("exact configuration/completed horizon required")
    _receipt(packet["initialization"], cohort=packet["cohort_id"], config=config, sources=sources)
    return config


def _failed_entry(owner, primary):
    try:
        owner.close()
    except BaseException as cleanup:  # noqa: BLE001 - preserve original entry and pending cleanup failures
        failure = BaseExceptionGroup("p-h3-image entry and cleanup failed", [primary, cleanup])
        failure.cleanup_owner = owner
        raise failure from primary


class Training(ownership.Owner):
    def __init__(self, host, cache, directory, *, mode, cohort_id, config, identity, owner_admission, resume=None, ema_decay=None):
        if mode not in ("fresh", "resume") or not isinstance(cohort_id, str) or not cohort_id or ema_decay is not None:
            raise ValueError("explicit fresh/resume/new cohort/noEMA intent required before components")
        if (mode == "fresh" and resume is not None) or (mode == "resume" and resume is None):
            raise ValueError("fresh has no restore; resume requires typed entire FULL")
        if configuration(**config) != config:
            raise ValueError("exact original128 configuration required")
        ownership.guard()
        validate_identity(identity)
        sources = ownership.sources()
        packet = read_packet(resume) if mode == "resume" else None
        if packet is not None:
            restored = _metadata(packet, FULL, identity=identity, host=host, sources=sources)
            if restored != config or packet["cohort_id"] != cohort_id or packet["game"].get("completed_steps") != packet["completed_steps"]:
                raise ValueError("whole FULL cohort/config/step differs")
        prepared, cache_identity = load_cache(cache, host=host, identity=identity)
        if packet is not None and packet["cache_identity"] != cache_identity:
            raise ValueError("original completed cache identity differs on strict resume")
        _admit(owner_admission, operation=mode, cohort=cohort_id, identity=identity, cache_identity=cache_identity, completed_steps=0 if packet is None else packet["completed_steps"])
        if Path(directory).exists():
            raise ValueError("new owned destination only; never replay an existing directory")
        super().__init__(host, mode=mode)
        self.identity, self.cohort_id, self.config = deepcopy(identity), cohort_id, deepcopy(config)
        self.sources, self.cache_identity, self.admission = sources, cache_identity, owner_admission
        self._initialization = None
        self.directory = Path(directory).absolute()
        devices = [host.device.index or 0] if host.device.type == "cuda" else []
        try:
            from ph3_research095.bridge import ConditionalCritic, conditions, coordinate_scale

            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
                if devices:
                    torch.cuda.set_rng_state(torch.Generator(device=host.device).manual_seed(config["seed"]).get_state(), host.device)
                self.attach()
                condition = conditions(prepared, len(host.timesteps), host.device)
                self.critic = ConditionalCritic(prepared[0].target.shape[1], condition.shape[1], device=host.device)
                self._initialization = policy.initialize(self, config, optimizers=()) if mode == "fresh" else deepcopy(packet["initialization"])
                new_directory(self.directory)
                self.initialization_record = write_json(self.directory / "initialization.json", self._initialization)
                self.original_backend = self.backend
                self.backend = ReceiptBackend(self.original_backend, self._initialization, sources, cache_identity)
                reference = get_recipe("bispf_optimized")
                if PG_VERSION != "0.9.5" or PG_SOURCE_PROFILE != "release-0.9.5":
                    raise ValueError("original official PG095 engine profile required")
                self.game = ConditionalPG095(
                    self.backend, prepared, self.critic, conditions=condition, particles=self.backend.network.particles,
                    response_domain="continuous_vector", response_scale=coordinate_scale(prepared),
                    sigma=reference.observation_sigma, batch_size=2, seed=config["seed"], total_steps=128,
                    recipe_overrides={k: config[k] for k in ("lr", "d_lr_mult", "prior_lr_mult")},
                    pg_source_profile="release-0.9.5", penalty_input_domain="continuous",
                )
                if self.game.prior.z is not self.adapter.particles or self.game.recipe.observation_sigma != reference.observation_sigma or self.game.sigma != reference.observation_sigma:
                    raise RuntimeError("original complete-bank ownership/public observation noise differs")
                if packet is not None:
                    self.game.load_state_dict(packet["game"])
                    if not equal(self.game.state_dict(), packet["game"]):
                        raise RuntimeError("entire typed game/optimizers/streams/RNG equality failed")
            self.entry_FULL = self.save()
            self.entry_RAW = self.export(self.directory / ("RAW" + str(self.step) + ".pt"))
        except BaseException as primary:
            _failed_entry(self, primary)
            raise

    @property
    def step(self):
        return self.game.completed_steps

    def update(self):
        self.require_open()
        _admit(self.admission, operation="update", cohort=self.cohort_id, identity=self.identity, cache_identity=self.cache_identity, completed_steps=self.step)
        return self.game.step()

    def _common(self, schema):
        return {
            "schema": schema, "profile": PROFILE, "cohort_id": self.cohort_id,
            "config": deepcopy(self.config), "sources": deepcopy(self.sources),
            "identity": deepcopy(self.identity), "host": host_identity(self.host),
            "cache_identity": deepcopy(self.cache_identity), "initialization": deepcopy(self._initialization),
            "selection": "raw", "ema_decay": None, "completed_steps": self.step,
        }

    def save(self):
        self.require_open()
        return save_packet(self.directory / ("FULL" + str(self.step) + ".pt"), {**self._common(FULL), "game": self.game.state_dict()})

    def export(self, path):
        self.require_open()
        exported = self.game.export_state_dict()
        return save_packet(path, {**self._common(RAW), "network": exported["network"], "backend_metadata": self.backend.metadata()})


class Inference(ownership.Owner):
    def __init__(self, host, *, raw, identity, owner_admission, ema_decay=None):
        if ema_decay is not None:
            raise ValueError("RAW inference forbids weight EMA")
        ownership.guard()
        validate_identity(identity)
        sources = ownership.sources()
        packet = read_packet(raw)
        config = _metadata(packet, RAW, identity=identity, host=host, sources=sources)
        _admit(owner_admission, operation="strict-RAW-infer", cohort=packet["cohort_id"], identity=identity, cache_identity=packet["cache_identity"], completed_steps=packet["completed_steps"])
        super().__init__(host, mode="inference")
        self.identity, self.packet, self.admission = deepcopy(identity), packet, owner_admission
        devices = [host.device.index or 0] if host.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(config["seed"]).get_state())
                if devices:
                    torch.cuda.set_rng_state(torch.Generator(device=host.device).manual_seed(config["seed"]).get_state(), host.device)
                self.attach()
                original = self.backend
                self.backend = ReceiptBackend(original, packet["initialization"], sources, packet["cache_identity"])
                if self.backend.metadata() != packet["backend_metadata"]:
                    raise ValueError("original native RAW backend identity differs")
                current, state = self.backend.network.state_dict(), packet["network"]
                if not isinstance(state, dict) or set(current) != set(state):
                    raise ValueError("strict original RAW parameter inventory required")
                for name, value in state.items():
                    if not isinstance(value, torch.Tensor) or value.dtype != current[name].dtype or value.shape != current[name].shape or not torch.isfinite(value).all():
                        raise ValueError("RAW tensor metadata/finite state differs")
                self.backend.network.load_state_dict(state, strict=True)
                if not equal(self.backend.network.state_dict(), state):
                    raise RuntimeError("strict original RAW tensor equality failed")
        except BaseException as primary:
            _failed_entry(self, primary)
            raise

    @torch.no_grad()
    def response(self, row, *, strength=1.0):
        self.require_open()
        validate_row(row, self.host)
        _admit(self.admission, operation="native-packed-response", cohort=self.packet["cohort_id"], identity=self.identity, cache_identity=self.packet["cache_identity"], completed_steps=self.packet["completed_steps"])
        # Original response validates [0,1] and keeps both image/audio velocity.
        return self.backend.response(row, strength=strength).detach().clone()
