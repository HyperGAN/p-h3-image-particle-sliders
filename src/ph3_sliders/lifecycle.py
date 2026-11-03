"""Register with the shared lifecycle, refusing live work before preparation."""

from particle_sliders.lifecycle import Backend, Capabilities

from .contract import REFUSAL, refuse_live


def validate(options):
    refuse_live("validate native configuration")


def prepare(options, **staged):
    refuse_live("prepare native model")


BACKEND = Backend(
    name="ph3-image",
    capabilities=Capabilities(
        generator_arm="routed_particle_gan",
        render=False,
        operations=(),
        base_precisions=(),
        render_outputs=(),
        qualification="Planning registration only; no native factory or model qualification",
        unavailable_reasons=tuple((op, REFUSAL) for op in (
            "train", "resume", "evaluate", "export", "response", "render")),
    ),
    prepare=prepare,
    validate=validate,
)
