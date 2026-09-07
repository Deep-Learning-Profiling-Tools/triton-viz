# Experimental Trn2 BF16 Tensor geometry term

Status: candidate, available through an explicit pipeline fit option. Independent hardware validation
is pending. This is not a measured improvement in the 254-case holdout MAPE.

The unchanged source-geometry fitter fails its original 20% control-suite CV gate
on the Trn2 cold collection: BF16 mean TensorE WAPE 26.8772%, FP32 3.7777%.
All 36 BF16 shapes have three compilations; median max/min spread is 1.02650,
with worst spread 1.13987. This suggests model mismatch rather than a single
extreme compilation. The hardware/compiler mechanism is not established.

For source tile counts D=mt*nt*kt, L=mt*kt, R=nt*kt, the experimental feature is
L*R*(1-L/D) = mt*(nt-1)*kt^2. It models a possible interaction between repeated
K-tile work and additional N tiles. It is zero for one N tile. It uses no target
compiler artifacts. The coefficient is fitted by NNLS on controls only.

Exploration on the existing controls compared the legacy basis with one extra
term at a time: D*R (17.44% CV), L*R (34.94%), R^2 (24.67%), and L*R*(1-L/D)
(6.31%). These are exploratory scores after feature selection, not independent
validation evidence. All original control points and the original CV gate remain.

## Compatibility

The fitter option `--bf16-revisit-term` is opt-in. Without it the original
five-feature fitter and CSV format remain unchanged. The pipeline passes
this option only when explicitly requested. Legacy CSV prediction arithmetic is unchanged and tested for exact
equality. New artifacts explicitly identify `tiled-revisit-v1`, carry the extra
coefficient and observed feature bounds, and reject missing/unknown versions.
Predictions outside the new feature bounds return `source_geometry_revisit_ood`.
These bounds describe marginal feature ranges, not proof of joint coverage.

FP32 retains the original five-feature fit, even when the option is enabled.
No hardware detection or Inf2 defaults
are changed. A control-data comparison of original and modified default fitters
produced byte-identical CSV and CV JSON. Inf2 profile data is unavailable here;
the 8.83% Inf2 end-to-end result has not been remeasured.

## Independent check before integration

Freeze the candidate coefficients before collecting
`microbench/inf2_nki/configs/trn2_revisit_independent_validation_v1.json`.
The config contains 18 unseen BF16 control shapes, three compilation trials each.
Use median TensorE active time for each shape. Require all 18 shapes, report OOD,
and require overall WAPE and each N-group WAPE strictly below 20%. Do not refit
on these labels or discard failing shapes. Keep hardware measurement serial
with the original cold collector.

Even a successful independent check does not validate the other calibration
gates or whole-program MAPE. Promotion of the experimental model remains pending that check. Replay CSVs
expose `tensor_source_geometry_ood_count`; the existing FLOP-domain counter
retains its separate meaning. The original calibration/provenance
manifests must not be relabeled as produced by the modified source revision.

## Running from the main checkout

```bash
python -m triton_viz.tools.nki_cost_model_pipeline fit --root "$ROOT" --bf16-revisit-term
python -m triton_viz.tools.nki_cost_model_pipeline evaluate --root "$ROOT"
```

The flag is valid only for fit. It retains every existing control CV gate.
Evaluate loads the versioned frozen calibration; no extra evaluation flag or
host-dependent model selection is needed. Omit the flag for the original Inf2
fit behavior.
