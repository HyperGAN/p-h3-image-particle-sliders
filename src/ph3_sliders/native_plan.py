"""Frozen inputs for the warm-lighting native LR experiment."""

from .native_host import SIGMAS


def experiment_plan():
    return dict(schema="ph3-native-experiment-v1", concept="warm golden lighting",
        model_revision="7f98acef503cf2bcfba821a78cb1cf1a9928b2df",
        resolution=dict(height=2048, width=2048), sigmas=list(SIGMAS),
        conditioning=dict(prompt_upsampler=False, mode="verbatim_native_text2image",
            neutral_suffix="The scene is illuminated by soft neutral daylight.",
            positive_suffix="The scene is illuminated by warm golden light.", references=[]),
        splits=dict(
            train=dict(seeds=[17, 29], scenes=[
                dict(id="cup", text="A documentary photograph of a ceramic cup on a wooden table."),
                dict(id="cottage", text="A documentary photograph of a stone cottage beside a field."),
                dict(id="trail", text="A documentary photograph of a wooded trail with ferns."),
                dict(id="fabric", text="A documentary photograph of folded linen beside a woven basket.")]),
            development=dict(seeds=[171, 172], scenes=[
                dict(id="bottle", text="A documentary photograph of a glass bottle on a shelf."),
                dict(id="bicycle", text="A documentary photograph of a bicycle parked on a city street."),
                dict(id="room", text="A documentary photograph of a reading room with books and an armchair."),
                dict(id="boat", text="A documentary photograph of a fishing boat moored at a dock.")]),
            confirmation=dict(seeds=[271, 272], scenes=[
                dict(id="bird", text="A documentary photograph of a small bird perched on a branch."),
                dict(id="bench", text="A documentary photograph of a wooden bench in a garden."),
                dict(id="lake", text="A documentary photograph of a mountain lake with a rocky shore."),
                dict(id="portrait", text="A documentary photograph of an adult standing beside a window.")])),
        training=dict(generator_rates=[6e-6, 2e-5, 6e-5], critic_rate=9e-4, particle_rate=6e-3,
            initial_seed=17, confirmation_seed=29, batch=8, microbatch=1, schedule_horizon=1600,
            ema=.995, common_selection_step=128, finalist_confirmation_step=256,
            final_minimum_steps=512, checkpoint_interval=64, metric_checkpoints=[32, 64, 128],
            inference_selection="ema", auxiliary_losses=[]),
        selection=dict(metric="matched_example_teacher_student_relative_l2_v2", lower_is_better=True,
            coordinate_system="full_packed_image_then_audio_velocity_unwhitened",
            complete_example_position_grid=True, require_all_scenes_below_OFF=True,
            require_positive_gain=True, require_exact_OFF=True, require_matched_media=True,
            tie_tolerance=1e-6, tie_breaker="lower_generator_lr", edge_winner="targeted_followup",
            raw_weights="diagnostic_only", failure="unscoreable"),
        samples=dict(strengths=[0., .25, .5, .75, 1.], reference="native_positive_conditioned_teacher",
            same_noise=True, same_sigmas=True, same_decoder=True),
        numerical_cohort=dict(frozen_base="bitsandbytes_int8", threshold=6.,
            bitsandbytes="0.50.2", activation_quantization_cast="float16",
            remaining_base_dtype="bfloat16_with_upstream_fp32_modules", adapter_dtype="float32",
            deterministic=True, tf32=False))


def prompt_map(plan):
    prompts = {}
    for split in plan["splits"].values():
        for scene in split["scenes"]:
            for kind in ("neutral", "positive"):
                name = scene["id"] + ":" + kind
                if name in prompts:
                    raise ValueError("scene IDs must be disjoint between experiment splits")
                prompts[name] = scene["text"] + " " + plan["conditioning"][kind + "_suffix"]
    return prompts


def expected_examples(plan, split):
    return [(scene["id"], seed, path) for scene in plan["splits"][split]["scenes"]
            for seed in plan["splits"][split]["seeds"] for path in ("neutral", "positive")]
