# Results overview

These are model-qualified electrical charge-equivalent results. Negative carriers mean the net electrical equivalent of electrons plus negative ions; the waveforms do not separate species. `—` means the metric did not clear its reporting gate.

Each result cell is `estimate [95% joint technical/calibration interval]; model sensitivity range`. The model range is usually the dominant uncertainty.

## Raw Q–V energy, apparent reactor power, and detected duty fraction

These electrical quantities do not require the passive-background or Cd model. Power is raw Qm–V loop power for the complete reactor circuit, not plasma-only power.

| Condition | Energy per duty period (µJ) | Apparent power (W) | Detected activity-on fraction |
|---|---:|---:|---:|
| 5 mM Mn nitrate, 4 kHz | 3.71e+03 | 14.9 | 20.1% |
| 5 mM Mn nitrate, 10 kHz | 2.44e+03 | 24.4 | 33.2% |
| 5 mM Mn nitrate, 20 kHz | 3.61e+03 | 72.2 | 48.3% |
| BMIM nitrate, 4 kHz | 3e+03 | 12 | 18.3% |
| BMIM nitrate, 10 kHz | 3.06e+03 | 30.6 | 30.1% |
| BMIM nitrate, 20 kHz | 3.25e+03 | 65 | 47.7% |
| Argon / no liquid, 4 kHz | 7.9e+03 | 31.6 | 42.2% |
| Argon / no liquid, 10 kHz | 6.56e+03 | 65.6 | 44.2% |
| Argon / no liquid, 20 kHz | 1.78e+03 | 35.5 | 42.5% |
| Pure water, 4 kHz | 9.23e+03 | 36.9 | 39.4% |
| Pure water, 10 kHz | 7.39e+03 | 74 | 45.3% |
| Pure water, 20 kHz | 6.42e+03 | 128 | 53.3% |

## Polarity-resolved charge-transfer estimates

| Condition | Polarity | Peak half-cycle (nC) | Peak half-cycle-average rate (e s⁻¹) | Whole-record rate (e s⁻¹) | Evidence/status |
|---|---:|---:|---:|---:|---|
| 5 mM Mn nitrate, 4 kHz | negative | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| 5 mM Mn nitrate, 4 kHz | positive | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| 5 mM Mn nitrate, 10 kHz | negative | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| 5 mM Mn nitrate, 10 kHz | positive | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| 5 mM Mn nitrate, 20 kHz | negative | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| 5 mM Mn nitrate, 20 kHz | positive | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| BMIM nitrate, 4 kHz | negative | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| BMIM nitrate, 4 kHz | positive | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| BMIM nitrate, 10 kHz | negative | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| BMIM nitrate, 10 kHz | positive | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| BMIM nitrate, 20 kHz | negative | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| BMIM nitrate, 20 kHz | positive | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Argon / no liquid, 4 kHz | negative | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Argon / no liquid, 4 kHz | positive | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Argon / no liquid, 10 kHz | negative | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Argon / no liquid, 10 kHz | positive | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Argon / no liquid, 20 kHz | negative | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Argon / no liquid, 20 kHz | positive | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Pure water, 4 kHz | negative | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Pure water, 4 kHz | positive | — | — | — | diagnostic_complex_background_model_rejected; one_or_more_metrics_not_resolved |
| Pure water, 10 kHz | negative | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| Pure water, 10 kHz | positive | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| Pure water, 20 kHz | negative | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |
| Pure water, 20 kHz | positive | — | — | — | diagnostic_transferred_background; one_or_more_metrics_not_resolved |

Bracketed intervals are 95% joint passive-calibration/MAX-capture moving-block bootstrap intervals. The explicitly displayed model-sensitivity ranges separately include the declared monitor scale, its shared effect on Ccell, a C′-to-|C*| scalar-model bracket, and the full-base Pyrex geometry scenario where used.

Peak aggregation is nested: within each capture and polarity, the maximum-amplitude carrier lobe is selected once per duty burst and its p95 is calculated; the condition estimate is then the median of those capture-level p95 values. Lobes are never pooled across the 64 captures to calculate a single p95.

A true instantaneous microdischarge peak is not resolved by the existing sampling. Retained surface charge is not inferred from polarity imbalance; it requires quiet pre/post-discharge plateaus.
