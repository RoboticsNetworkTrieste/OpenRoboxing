# LICENSING — what may be published, and on what terms

Task `M6-T2`. `WORKPLAN` marks it **blocking**: *"Publishing finetuned derivatives of
NVIDIA-licensed checkpoints is redistribution of derived models. Confirm the terms, produce the
attribution text, and document the base checkpoint and job config for every published weight set."*

Decision taken by the project owner (Carlo Dentesano) on **2026-08-08**: weights may be published.

This page records the terms that make that decision correct, so nobody has to take it on trust.

---

## The terms

The upstream repository is **dual-licensed**, and its own `LICENSE` file at the repository root says
so explicitly:

| Component | Licence |
|---|---|
| Source code, scripts, software | **Apache License 2.0** |
| Trained model checkpoints and weights | **NVIDIA Open Model License** (last modified 2025-10-24) |

The NVIDIA Open Model License is **permissive about exactly the thing M6-T2 was worried about**.
Section 2.2 grants a "perpetual, worldwide, non-exclusive, no-charge, royalty-free, revocable
license to … create derivative works of … distribute and import the Model", and Section 2.4 states
plainly: *"You own Your Derivative Models."*

So a finetuned GEAR-SONIC checkpoint is ours to publish. The blocker was real — it just resolves in
our favour once read.

## What we must do when we publish weights

Section 3 imposes three conditions, and only three:

- **(a) Ship a copy of the Agreement.** Include the upstream `LICENSE` file alongside any published
  weight set.
- **(b) Carry the attribution notice**, verbatim:

  > Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

  It may sit wherever other third-party notices live.
- **(c) Cosmos models only** — a phrase "Built on NVIDIA Cosmos". **See the open question below.**

Section 2.1 adds two ways to *lose* the licence, both of which we should simply not do: initiating
patent or copyright litigation over the Model, and circumventing built-in safety guardrails without
comparable alternatives. Neither is in scope for a boxing game.

Section 2.3 requires use in accordance with NVIDIA's Trustworthy AI terms.

## ⚠ One open question, and it is small

Condition (c) applies to a "**NVIDIA Cosmos Model**", defined in §1.4 as "a multimodal Model that is
covered by this Agreement". **GEAR-SONIC is a whole-body control policy, not a multimodal model**, so
on a plain reading (c) does not apply and the phrase "Built on NVIDIA Cosmos" would be inaccurate
rather than merely redundant.

That is my reading, not a lawyer's. It costs nothing to check with NVIDIA before the first public
weight release, and it is the only point on this page where the answer is not simply written down.

## Attribution text to ship

Put this in the release notes, the model card, and any page offering a download:

```
This model is a derivative of GEAR-SONIC from NVlabs/GR00T-WholeBodyControl.
Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

Base checkpoint : gear_sonic_deploy/policy/release
                  model_encoder.onnx  sha256 013ab0287236aa27...
                  model_decoder.onnx  sha256 c7241a123eaa36b5...
Robot model     : g1_29dof_old.xml    sha256 58660a6f1d0d33ff...

OpenRoboxing source code is Apache-2.0, as is the upstream source it builds on.
```

The exact hashes for any given release come from that release's season manifest
(`league/manifest.py`); the ones above are illustrative and will differ per freeze.

## Per-weight-set documentation

M6-T2 also requires "the base checkpoint and job config for every published weight set". The
machinery exists and is not optional:

- `tools/freeze_season.py` pins every asset by SHA-256, including both ONNX files and the robot
  model. **A manifest is the release record.**
- A finetuned weight set additionally needs its **job config**, which is `S-T2`'s to produce. Until
  S-T2 exists there are no finetuned weights to publish — only upstream's, unmodified, which this
  page already covers.

## The gate in code

`league/manifest.py` refuses to mark a manifest `released` unless the caller passes an exact
acknowledgement string. That gate stays, deliberately: this page makes the answer *knowable*, and the
gate makes publishing a *deliberate act* rather than a default. Release a manifest with:

```bash
python -m openroboxing.tools.freeze_season \
    --season season-0 --at <iso8601> \
    --released M6-T2-signed-off \
    --out seasons/season-0.json
```

## Status

| | |
|---|---|
| Terms confirmed | ✅ NVIDIA Open Model License permits derivative models and their distribution |
| Attribution text produced | ✅ above |
| Human sign-off | ✅ Carlo Dentesano, 2026-08-08 |
| Cosmos clause checked with NVIDIA | ⬜ open, low risk (see above) |
| Finetuned weight sets documented | n/a — none exist yet (`S-T2`) |

**Nothing has been published.** This page and the release gate say what *may* be; pushing files
anywhere is a separate, deliberate act by a human.
