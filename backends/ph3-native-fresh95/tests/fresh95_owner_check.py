"""FOUR NEW UNRUN software cases; one original PG095 update in case3 only.

The synthetic frozen host exercises the original H3Host packed image/audio
protocol, RoutedAdapter/HNDL/NativeResponse and original GlobalMixCritic.
It is not a pretrained Diffusers/model/cache/native-render qualification.
"""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from particle_sliders.pg09_backends import PreparedResponse
from ph3_research095.bridge import NativeResponse, NativeRow
from ph3_research095.native_host import H3Host

from ph3_native_fresh95 import configuration, open_inference, open_training, save_prepared_cache
from ph3_native_fresh95 import api, ownership, policy
from ph3_native_fresh95.cache import prepared_digest
from ph3_native_fresh95.util import equal


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q, self.to_k, self.to_v = (nn.Linear(4, 4) for _ in range(3))
        self.to_out = nn.Sequential(nn.Linear(4, 4))

    def forward(self, value):
        return self.to_out(torch.tanh(self.to_q(value) + self.to_k(value) + self.to_v(value)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.gradient_checkpointing = False

    def forward(self, value):
        return value + self.attn(value)


def original_checkpoint(function, *args, **kwargs):
    return function(*args, **kwargs)


class SyntheticPackedHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_layers=1, num_refiner_layers=1, hidden_size=4,
            num_attention_heads=1, attention_head_dim=4, in_channels=4, audio_in_channels=4,
            text_dim=4, patch_size=(1, 1, 1))
        self.transformer_blocks = nn.ModuleList([Block()])
        self.token_refiner = nn.Module()
        self.token_refiner.refiner_blocks = nn.ModuleList([Block()])
        self.enable_gradient_checkpointing(gradient_checkpointing_func=original_checkpoint)

    @property
    def is_gradient_checkpointing(self):
        return any(getattr(m, "gradient_checkpointing", False) for m in self.modules())

    def enable_gradient_checkpointing(self, *, gradient_checkpointing_func):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = gradient_checkpointing_func

    def forward(self, *, hidden_states, audio_hidden_states, encoder_hidden_states, timestep,
                timestep_indices, return_dict, position_ids, token_tags, video_indices, audio_indices,
                text_indices):
        assert return_dict is False and hidden_states.shape == audio_hidden_states.shape == (1, 4, 4)
        assert position_ids is not None and token_tags is not None and timestep is not None
        assert timestep_indices is not None and video_indices.numel() == audio_indices.numel() == 4
        assert text_indices.numel() == encoder_hidden_states.shape[1]
        context = encoder_hidden_states.mean(1, keepdim=True)
        image, audio = hidden_states + context, audio_hidden_states + context
        for block in tuple(self.token_refiner.refiner_blocks) + tuple(self.transformer_blocks):
            if torch.is_grad_enabled() and block.gradient_checkpointing:
                image = block._gradient_checkpointing_func(block, image)
                audio = block._gradient_checkpointing_func(block, audio)
            else:
                image, audio = block(image), block(audio)
        return image, audio


def host_fixture():
    with torch.random.fork_rng():
        torch.manual_seed(17)
        return H3Host(SyntheticPackedHost().eval().requires_grad_(False), latent_height=2, latent_width=2)


def config_fixture():
    return configuration(lr=0.004, d_lr_mult=4.5, prior_lr_mult=7.5, seed=7, ema_decay=None)


def identity_fixture():
    return {"kind": "NEW-synthetic-host-protocol-fixture", "model": {"fixture_seed": 17, "not_pretrained": True}, "completed_owner_receipt": None}


def admitted(request):
    assert request["ema_decay"] is None and request["selection"] == "raw"
    assert request["stop_on_established_qualified_own_failure_or_owner_STOP"] is True
    return True


def cache_fixture(tmp_path, host, config):
    neutral = host.conditioning("NEW synthetic neutral", torch.full((1, 3, 4), 0.1))
    positive = host.conditioning("NEW synthetic positive", torch.full((1, 3, 4), 0.3))
    prepared = []
    for i, path in enumerate(("neutral", "positive")):
        row = NativeRow(host.initial_state(31), neutral, positive, "NEW software scene", 31, path,
            i, float(host.timesteps[i]))
        prepared.append(PreparedResponse(row, torch.full((1, 32), 0.2 + i * 0.1), torch.zeros(1, 32)))
    origins = [{"row_index": i, "completed": True, "kind": "NEW-synthetic-software-data",
        "prepared_sha256": prepared_digest(item), "completed_output_receipt": None}
        for i, item in enumerate(prepared)]
    binding = save_prepared_cache(tmp_path / "completed.pt", host=host, identity=identity_fixture(),
        config=config, prepared=prepared, teacher_origins=origins, cache_id="NEW software rows only")
    return binding, prepared


def forbid_teachers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cached API attempted teacher/prepare/rollout")
    monkeypatch.setattr(NativeResponse, "prepare", forbidden)
    monkeypatch.setattr(H3Host, "collect_pair", forbidden)
    monkeypatch.setattr(H3Host, "rollout", forbidden)


def open_fresh(host, cache, path, *, cohort="NEW-p-h3-software"):
    return open_training(host, cache, path, mode="fresh", cohort_id=cohort, config=config_fixture(),
        identity=identity_fixture(), owner_admission=admitted)


def test_new_original_hostlaw_Parameters_zeroUP_full_coupled_velocity_and_receipt_before_optimizers(tmp_path, monkeypatch):
    host, config = host_fixture(), config_fixture()
    cache, prepared = cache_fixture(tmp_path, host, config)
    forbid_teachers(monkeypatch)
    base = [(p, p.detach().clone()) for p in host.model.parameters()]
    initialize, game = policy.initialize, api.ConditionalPG095
    seen = {}
    def init_spy(owner, config, **kwargs):
        before = deepcopy(owner.critic.state_dict())
        ids = [id(p) for p in owner.adapter.parameters()] + [id(p) for p in owner.critic.parameters()]
        receipt = initialize(owner, config, **kwargs)
        assert equal(before, owner.critic.state_dict())
        assert ids == [id(p) for p in owner.adapter.parameters()] + [id(p) for p in owner.critic.parameters()]
        seen["receipt"] = receipt
        return receipt
    def game_spy(backend, prepared, critic, **kwargs):
        assert backend.metadata()["initialization"] == seen["receipt"]
        assert json.loads((tmp_path / "fresh" / "initialization.json").read_text()) == seen["receipt"]
        assert kwargs["particles"] is backend.network.particles is backend.original.adapter.particles
        seen["before_optimizers"] = True
        return game(backend, prepared, critic, **kwargs)
    monkeypatch.setattr(policy, "initialize", init_spy)
    monkeypatch.setattr(api, "ConditionalPG095", game_spy)
    owner = open_fresh(host, cache, tmp_path / "fresh")
    try:
        assert seen["before_optimizers"] and owner.step == 0 and owner.entry_FULL.bytes > 0
        assert owner.game.prior.z is owner.adapter.particles and owner.adapter.particles.shape == (128, 4)
        assert all(torch.count_nonzero(b.graph["up"].weight).item() == 0 for b in owner.adapter.branches)
        generated = owner.original_backend.generated(prepared[0])
        with owner.adapter.scaled(0):
            expected = host.response(host.velocity(prepared[0].row.state, prepared[0].row.neutral, 0))
        assert generated.shape == (1, 32) and torch.equal(generated, expected)
        assert all(p is actual and not p.requires_grad and torch.equal(p, value)
            for (p, value), actual in zip(base, host.model.parameters()))
        assert "constructor KEEP" in owner._initialization["critic_policy"]
    finally:
        owner.close()
    assert host.adapter is None and all(not m._forward_hooks for m in host.model.modules())


def test_new_deterministic_sqrt5_QR_std1_R2_bank_router_critic_KEEP_and_ambient_RNG(tmp_path):
    first_host, config = host_fixture(), config_fixture()
    cache, _ = cache_fixture(tmp_path, first_host, config)
    before = torch.get_rng_state().clone()
    first = open_fresh(first_host, cache, tmp_path / "a", cohort="fresh-a")
    second = None
    try:
        assert torch.equal(before, torch.get_rng_state())
        second = open_fresh(host_fixture(), cache, tmp_path / "b", cohort="fresh-b")
        assert equal(first.adapter.state_dict(), second.adapter.state_dict())
        assert equal(first.critic.state_dict(), second.critic.state_dict())
        for branch in first.adapter.branches:
            weight = branch.graph["down"].weight.detach()
            gram = weight @ weight.T if weight.shape[0] <= weight.shape[1] else weight.T @ weight
            expected = torch.eye(min(weight.shape)) * max(weight.shape) / (3 * branch.graph["down"].in_features)
            assert torch.allclose(gram, expected, atol=2e-6, rtol=2e-5)
        policies = first._initialization["parameter_policies"]
        bank = next(r for r in policies if r["role"] == "bank-R2Normal-std1")
        assert bank["before_sha256"] != bank["after_sha256"] and bank["distribution"] == "R2Normal(0,1)"
        assert abs(float(first.adapter.particles.detach().std(unbiased=False)) - 1) < 0.1
        assert all(r["before_sha256"] == r["after_sha256"] for r in policies
            if r["role"] in ("original-constructor-KEEP", "protected-zero-UP"))
    finally:
        first.close()
        if second is not None:
            second.close()


def test_new_ONE_original_update_typed_whole_FULL_optimizer_RNG_and_RAW_skip_init_cache_critic(tmp_path, monkeypatch):
    host, config = host_fixture(), config_fixture()
    cache, prepared = cache_fixture(tmp_path, host, config)
    forbid_teachers(monkeypatch)
    first = open_fresh(host, cache, tmp_path / "first", cohort="resume")
    try:
        first.update()  # The ONLY successful update across all FOUR new cases.
        full, raw = first.save(), first.export(tmp_path / "RAW1.pt")
        saved, receipt = deepcopy(first.game.state_dict()), deepcopy(first._initialization)
        assert first.step == 1
    finally:
        first.close()
    def forbidden(*args, **kwargs):
        raise AssertionError("resume/RAW attempted fresh initialization or extra component")
    monkeypatch.setattr(policy, "initialize", forbidden)
    resumed = open_training(host_fixture(), cache, tmp_path / "resumed", mode="resume", cohort_id="resume",
        config=config, identity=identity_fixture(), owner_admission=admitted, resume=full)
    try:
        assert resumed.step == 1 and equal(saved, resumed.game.state_dict()) and resumed._initialization == receipt
    finally:
        resumed.close()
    monkeypatch.setattr(api, "ConditionalPG095", forbidden)
    monkeypatch.setattr(api, "load_cache", forbidden)
    inference = open_inference(host_fixture(), raw=raw, identity=identity_fixture(), owner_admission=admitted)
    try:
        assert not hasattr(inference, "game") and not hasattr(inference, "critic")
        assert equal(inference.backend.network.state_dict(), saved["network"])
        assert inference.response(prepared[0].row, strength=0.5).shape == (1, 32)
        for strength in (-1, 2, float("nan")):
            with pytest.raises(ValueError):
                inference.response(prepared[0].row, strength=strength)
    finally:
        inference.close()


def test_new_intent_cache_STOP_early_refusal_and_retryable_foreign_safe_hooks_flags_marker_LAST(tmp_path, monkeypatch):
    host, config = host_fixture(), config_fixture()
    cache, _ = cache_fixture(tmp_path, host, config)
    forbid_teachers(monkeypatch)
    for options in ({"mode": "bad"}, {"mode": "resume"}, {"mode": "fresh", "ema_decay": 0.5}):
        with pytest.raises(ValueError):
            open_training(host, cache, tmp_path / "refused", cohort_id="new", config=config,
                identity=identity_fixture(), owner_admission=admitted, **options)
    with pytest.raises(ValueError):
        open_training(host, replace(cache, checkpoint=replace(cache.checkpoint, sha256="0" * 64)),
            tmp_path / "bad-cache", mode="fresh", cohort_id="new", config=config,
            identity=identity_fixture(), owner_admission=admitted)
    with pytest.raises(ValueError):
        open_training(host, cache, tmp_path / "STOP", mode="fresh", cohort_id="new", config=config,
            identity=identity_fixture(), owner_admission=lambda request: False)
    assert host.adapter is None and not any(vars(m).get(ownership.MARKER) for m in host.model.modules())
    owner = open_fresh(host, cache, tmp_path / "cleanup")
    container, handle = owner.adapter.handles, owner.adapter.handles[0]
    remove, attempts = handle.remove, []
    def fail_once():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("NEW one-time cleanup fault")
        remove()
    monkeypatch.setattr(handle, "remove", fail_once)
    foreign = host.model.transformer_blocks[0].attn.to_q.register_forward_hook(lambda m, a, value: value)
    container.append(foreign)
    with pytest.raises(RuntimeError):
        owner.close()
    assert attempts == [] and foreign.id in host.model.transformer_blocks[0].attn.to_q._forward_hooks
    container.pop()
    with pytest.raises(BaseExceptionGroup):
        owner.close()
    assert owner.cleanup_pending and not owner.closed and owner.adapter.handles == [handle]
    assert all(vars(m).get(ownership.MARKER) is owner for m in host.model.modules())
    with pytest.raises(RuntimeError):
        owner.update()  # Refused before another ML call.
    owner.adapter.handles = [foreign]
    with pytest.raises(RuntimeError):
        owner.close()
    assert attempts == [1] and owner.adapter.handles == [foreign]
    owner.adapter.handles = container
    dictionary = handle.hooks_dict_ref()
    original, callback = dictionary[handle.id], lambda *args: None
    dictionary[handle.id] = callback
    with pytest.raises(RuntimeError):
        owner.close()
    assert dictionary[handle.id] is callback and attempts == [1]
    dictionary[handle.id] = original
    host.model.training = True
    with pytest.raises(RuntimeError):
        owner.close()
    assert host.model.training is True and attempts == [1]
    host.model.training = False
    original_setattr, flag_faults = Block.__setattr__, []
    def fail_one_flag_restore(module, name, value):
        if name == "_gradient_checkpointing_func" and value is original_checkpoint and not flag_faults:
            flag_faults.append(1)
            raise RuntimeError("NEW borrowed callback restoration fault")
        return original_setattr(module, name, value)
    monkeypatch.setattr(Block, "__setattr__", fail_one_flag_restore)
    with pytest.raises(BaseExceptionGroup):
        owner.close()
    assert flag_faults == [1] and owner.cleanup_pending and not owner.closed
    assert all(vars(m).get(ownership.MARKER) is owner for m in host.model.modules())
    original_delattr, marker_faults = SyntheticPackedHost.__delattr__, []
    def fail_one_marker_release(module, name):
        if name == ownership.MARKER and not marker_faults:
            marker_faults.append(1)
            raise RuntimeError("NEW marker-only release fault")
        return original_delattr(module, name)
    monkeypatch.setattr(SyntheticPackedHost, "__delattr__", fail_one_marker_release)
    with pytest.raises(BaseExceptionGroup):
        owner.close()
    assert marker_faults == [1] and owner.cleanup_pending and not owner.closed
    assert host.adapter is None and vars(host.model).get(ownership.MARKER) is owner
    host.model.training = True
    with pytest.raises(RuntimeError):
        owner.close()
    assert host.model.training is True
    host.model.training = False
    owner.close()
    owner.close()
    assert owner.closed and attempts == [1, 1] and host.adapter is None
    assert foreign.id in host.model.transformer_blocks[0].attn.to_q._forward_hooks
    assert all(ownership.MARKER not in vars(m) for m in host.model.modules())
    foreign.remove()  # The foreign owner alone removes its handle.
