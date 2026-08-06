# DBD surface-charge analysis

`DBD_Surface_Charge_Analysis.py` is the companion analysis for the voltage-scan archive. It reads the ZIP directly, so it does not extract or modify the PicoScope files.

## Run

From the repository directory in PowerShell:

```powershell
& "C:\Users\isaac\anaconda3\python.exe" .\DBD_Surface_Charge_Analysis.py `
  "C:\Users\isaac\OneDrive\Documents\McGill\McGill Experiments\waveforms\July 2026\Lissajouswaveformsdifferentmediums.zip" `
  --output-dir .\dbd_surface_charge_report
```

The defaults use 16 captures at each voltage-scan level, every legacy maximum-voltage capture, 3,000 capture-level bootstrap replicates, and both PNG and PDF figures.

For a quick diagnostic run:

```powershell
& "C:\Users\isaac\anaconda3\python.exe" .\DBD_Surface_Charge_Analysis.py `
  "C:\path\to\Lissajouswaveformsdifferentmediums.zip" `
  --output-dir .\dbd_surface_charge_quickcheck `
  --files-per-scan-level 2 --files-per-maximum 2 --bootstrap-replicates 200
```

## Main outputs

- `RESULTS_OVERVIEW.md`: compact human-readable result table and interpretation.
- `headline_results.csv`: one row per condition and charge polarity, including requested estimates, conditional repeatability intervals, joint calibration/capture analysis intervals, model-sensitivity ranges, and reporting status.
- `supervisor_summary.csv`: complete condition-level result table.
- `long_form_results.csv`: tidy result table for plotting or importing into another program.
- `capacitance_and_fit_results.csv`: whole-waveform complex `Ccell`, gated three-level effective `Cd`, model gates, and the separate geometry scenario.
- `per_file_maximum_metrics.csv`: capture-level results used by the bootstrap.
- `peak_halfcycle_observations.csv`: audit trail for each duty-burst peak half-cycle.
- `burst_period_observations.csv`: per-period raw Q–V shoelace energy, duty fraction, loop-closure diagnostics, and channel-specific clipping gates.
- `electrical_condition_metrics.csv`: condition-level raw energy, apparent power, and duty-fraction summaries.
- `dose_response_binned.csv` and `stationarity_binned.csv`: capture-balanced plot data.
- `stationarity_metrics.csv`: first-to-last record-quintile drift diagnostics.
- `dose_clock_results.csv`: ideal negative-charge-equivalent dose curves versus volume, including the 2.5 mL reactor point.
- `breakdown_level_fit_points.csv`: points used in the percentage-of-breakdown plots.
- `METHODS_AND_LIMITATIONS.md`: equations, definitions, and uncertainty rules generated with the run settings.
- `FIGURE_CAPTIONS.md`: ready-to-adapt captions for the figure set.
- `analysis_audit.json`: archive, random seed, file failures, and calibration decisions.

The `figures` directory contains:

1. A signal-processing example showing voltage, monitor charge, locked polarity, half-cycle boundaries, passive background, and excess charge.
2. Data-coverage and complex-`Ccell`/ADC-quantization audit summaries.
3. Positive/negative peak-half-cycle charge, peak half-cycle-average flow, and whole-record flow with separate repeatability and model-sensitivity intervals.
4. Retained-charge availability.
5. Q–V traces for every voltage scan and all maximum-voltage conditions.
6. Percentage-of-breakdown fits, including the gated 100/105/115% effective-`Cd` regression; legacy MAX remains out of sample.
7. Raw Q–V loop energy/apparent reactor power and detected duty-on fraction.
8. Capture-balanced per-lobe dose response and within-record stationarity.
9. Ideal negative-charge-equivalent dose clock versus volume.

## What the reported quantities mean

The script detects the duty-burst frequency from each waveform activity envelope and detects the carrier separately. It fits the full 40%/60%/75% passive waveforms to in-phase and quadrature carrier bases (`C* = C' − iC''`), including an offset and linear drift. Exact fitted carrier-basis changes at each lobe's two zero-crossing endpoints give the passive charge prediction. This replaces the lobe-amplitude ratio fit that could collapse onto an ADC-code ratio. Signals below eight peak-to-peak Channel-D codes remain explicitly quantization-limited. The unused 90% captures act as an out-of-sample passive-prediction check.

- **Positive/negative charge per peak half-cycle** uses nested aggregation: calculate p95 within each capture across the maximum-amplitude carrier lobe selected once per duty burst, then report the median of the capture-level p95 values. Lobes from all captures are not pooled.
- **Peak rate** is p95 of `q/(e Δt)` across the same maximum-amplitude lobe selected once per duty burst. It is a half-cycle-average net charge-equivalent rate, not an instantaneous nanosecond microdischarge peak.
- **Whole-record rate** is the signed background-subtracted charge sum divided by elementary charge and the full record duration, including duty-off time.
- **Retained charge** is reported only when quiet, stable plateaus exist at both record edges. Positive/negative transport imbalance is never relabeled as retained surface charge.
- **Apparent power** is the raw cyclic Qm–V shoelace area per complete duty-burst period multiplied by measured burst frequency. It includes plasma, dielectric, liquid, and phase-skew losses; it is not plasma-only power.
- **Duty-on fraction** is the fraction of each period above `P10 + 0.30(P90−P10)` in the selected activity envelope.
- **Dose clock** assumes 100% useful delivery of total negative charge equivalents. It is neither electron-specific nor a chemical-conversion prediction; BMIM is a hypothetical rate-transfer comparison, and Mn²⁺ needs at least two equivalents for full reduction.

“Negative carriers” means a net external-terminal negative electrical equivalent assigned to the plasma–liquid interface under the configured polarity mapping. By default, a negative pin-voltage lobe is assumed to mean negative-carrier delivery. Channel-D sign locking does not validate that physical assignment, and dielectric memory voltage can shift gas conduction relative to the source-voltage zero crossing. The waveform cannot separate electrons from negative ions or measure their gross counts independently. Likewise, the result is a total rate, not area-normalized flux, unless `--active-area-mm2` is supplied.

## Confidence and model status

`repeat_ci` resamples only the sequential MAX captures while holding calibration fixed. `analysis_ci` jointly resamples the 40%/60%/75% complex-background captures, the 100%/105%/115% active-`Cd` captures when that fit passes, and the MAX captures in moving blocks; carrier half-cycles are not treated as independent replicates. These describe technical repeatability, not a population of independently rebuilt experiments. The broader model-sensitivity ranges additionally propagate the declared 0.1 µF monitor-capacitor tolerance, Channel-D gain tolerance, the shared effect of that scale on `Ccell`, and keep the measured active-`Cd` and full-base Pyrex geometry as separate model scenarios.

The effective active-branch `Cd` fit uses the three independent 100%, 105%, and 115% level medians with an intercept. It must pass clipping, activity-at-breakdown, monotonicity, `R²`, pairwise-slope, capture-count, and `Cd > 1.05 Ccell` bootstrap gates. A passing result is labeled **provisional** because three levels do not replace a denser amplitude scan or direct dielectric measurement. The full-beaker-base Pyrex result is retained separately as an exploratory model scenario.

## Useful options

```text
--charge-polarity auto|1|-1
--target-negative-on-pin-negative / --no-target-negative-on-pin-negative
--active-area-mm2 VALUE
--reference-capacitance-relative-uncertainty VALUE
--monitor-gain-relative-uncertainty VALUE
--beaker-diameter-cm VALUE
--glass-thickness-mm VALUE
--pyrex-epsilon-min VALUE --pyrex-epsilon-max VALUE
--liquid-volume-ml VALUE --metal-ion-concentration-mM VALUE
--dose-electrons-per-metal-ion VALUE
--no-pdf --no-plots
```

Run `python DBD_Surface_Charge_Analysis.py --help` for the complete list.
