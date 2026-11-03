# Official license inspection — 2026-11-03

Inspected [PrunaAI/P-H3-Image's LICENSE](https://huggingface.co/PrunaAI/P-H3-Image/blob/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/LICENSE),
[NOTICE](https://huggingface.co/PrunaAI/P-H3-Image/blob/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/NOTICE)
and [model card](https://huggingface.co/PrunaAI/P-H3-Image/blob/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/README.md).
The license identifies itself as the MiniMax H3 Community License Agreement,
dated August 2, 2026. `backend.lock.json` records inspected document hashes.

The operative terms are broader than a restriction on downloading weights:

- I.3–I.5 define the granted territory by excluding the EU, UK, Republic of Korea
  and United States of America.
- I.6–I.11 include model software, parameters, inference-supporting code,
  documentation and model derivatives in the defined works/materials.
- II limits the grant to the applicable territory, while describing a separate
  licensing route for excluded territories.
- III conditions redistribution on territory, downstream license delivery,
  modification notices and the specified NOTICE text. III.3 encourages the
  attribution phrase and generation identifiers; it does not make every
  encouraged item an unconditional requirement.
- IV.1 requires separate prior written authorization above US$20 million yearly
  revenue from commercial products/services. IV.2 requires prominent MiniMax H3
  identification in a commercial product/service UI. The model card's attribution
  summary is shorter than these exact distinctions.
- V.2 requires protective downstream terms. V.3 restricts using the works or
  outputs to improve other AI models outside H3 and its derivatives. V.4 covers
  use, reproduction, modification, distribution and display of works and outputs
  outside the granted territory. V.5 adds safeguards for generation services.
- Exhibit A includes acceptable-use and public AI-content disclosure obligations.

Authorization for this task does not establish a separate MiniMax license.
No territory, qualifying commercial revenue status or separate license has been
asserted or inferred. This revision therefore executes none of those native
operations. The upstream Qwen components' Apache licenses do not relicense the
whole adaptation or remove the H3 restrictions.

Source-only repository creation is technically possible: this repository holds
original planning/registration glue and an unrelated random-Linear fixture that
depends on the separately licensed shared library. It does not redistribute H3
source, weights, native model configurations, learned derivatives or outputs.
The original-source Apache license is not a determination that a future H3
derivative or integration has permission. Keep source review and model licensing
separate, and resolve the model's rights before implementing or executing native
model work. This is a recorded engineering gate, not a legal clearance.

The official [MiniMax licensing form](https://platform.minimax.io/h3-license)
is the next licensing route. Any authorization must cover the intended territory,
adaptation, fine-tuning/LoRA work, intended outputs and downstream distribution;
an unrelated hosted-service entitlement is not assumed to cover local training.
