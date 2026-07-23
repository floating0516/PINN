# Gated Tangential Magnitude Residual Design

## Goal

Determine quickly whether tangential GNSS motion improves magnitude prediction when it cannot disturb the established radial STF and physics path.

## Controlled Experiment

- Use the same USGS-priority NPZ, accepted rows, station-random split, seed 42, loss, optimizer, scheduler, and 200 epochs as the R-only baseline.
- Load the formal R-only seed-42 checkpoint and freeze all radial/base parameters.
- Keep STF prediction entirely radial, so the radial waveform synthesis loss is unchanged.
- Encode T with a small independent temporal branch conditioned on the existing five metadata values.
- Predict `Mw = Mw_R + tanh(gate) * delta_Mw_T`, with `gate` initialized to exactly zero.
- Do not add Z, mechanism labels, new data filtering, or additional hyperparameter variants.

## Compatibility

The new behavior is enabled only by `model.input_fusion: magnitude_gated_residual` with `model.input_components: [radial, tangential]`. Existing radial and early-fusion R+T configurations keep their current architecture and checkpoint behavior.

## Verification And Stop Rule

Add one focused model test proving that a warm-started zero-gate model exactly reproduces R-only predictions while freezing the radial path and allowing the gate to receive a gradient. Run one CUDA forward/backward smoke and the full suite once before training.

Train only seed 42. Accept T only if all internal gates pass: station MAE `<= 0.0885`, event MAE `<= 0.1613`, and strike-slip event MAE `< 0.1801`. Evaluate the fixed eight external events once. If an internal gate fails, stop the T direction and do not train seeds 17/73.
