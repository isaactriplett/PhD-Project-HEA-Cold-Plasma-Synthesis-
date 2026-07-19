# Results overview

These are model-qualified electrical charge-equivalent results. Negative carriers mean the net electrical equivalent of electrons plus negative ions; the waveforms do not separate species. `—` means the metric did not clear its reporting gate.

Each result cell is `estimate [95% joint technical/calibration interval]; model sensitivity range`. The model range is usually the dominant uncertainty.

## Raw Q–V energy, apparent reactor power, and detected duty fraction

These electrical quantities do not require the passive-background or Cd model. Power is raw Qm–V loop power for the complete reactor circuit, not plasma-only power.

| Condition | Energy per duty period (µJ) | Apparent power (W) | Detected activity-on fraction |
|---|---:|---:|---:|
| 5 mM Mn nitrate, 4 kHz | 3.69e+03 | 14.8 | 19.0% |
| 5 mM Mn nitrate, 10 kHz | 2.53e+03 | 25.3 | 33.4% |
| 5 mM Mn nitrate, 20 kHz | 3.53e+03 | 70.7 | 48.2% |
| BMIM nitrate, 4 kHz | 3e+03 | 12 | 18.2% |
| BMIM nitrate, 10 kHz | 3.02e+03 | 30.2 | 30.4% |
| BMIM nitrate, 20 kHz | 3.24e+03 | 64.7 | 47.9% |
| Argon / no liquid, 4 kHz | 7.79e+03 | 31.2 | 42.0% |
| Argon / no liquid, 10 kHz | 7.16e+03 | 71.6 | 44.3% |
| Argon / no liquid, 20 kHz | 1.78e+03 | 35.5 | 41.8% |
| Pure water, 4 kHz | 9.13e+03 | 36.5 | 39.3% |
| Pure water, 10 kHz | 7.35e+03 | 73.5 | 45.6% |
| Pure water, 20 kHz | 6.4e+03 | 128 | 53.7% |

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
