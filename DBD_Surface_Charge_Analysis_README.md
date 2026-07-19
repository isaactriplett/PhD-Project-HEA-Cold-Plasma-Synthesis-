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
- `capacitance_and_fit_results.csv`: complex `Ccell`, diagnostic `Cd`, model gates, and geometry scenario.
- `per_file_maximum_metrics.csv`: capture-level results used by the bootstrap.
- `peak_halfcycle_observations.csv`: audit trail for each duty-burst peak half-cycle.
- `breakdown_level_fit_points.csv`: points used in the percentage-of-breakdown plots.
- `METHODS_AND_LIMITATIONS.md`: equations, definitions, and uncertainty rules generated with the run settings.
- `FIGURE_CAPTIONS.md`: ready-to-adapt captions for the figure set.
- `analysis_audit.json`: archive, random seed, file failures, and calibration decisions.

The `figures` directory contains:

1. A signal-processing example showing voltage, monitor charge, locked polarity, half-cycle boundaries, passive background, and excess charge.
2. Data-coverage and supported complex-`Ccell` summaries.
3. Positive/negative peak-half-cycle charge, peak half-cycle-average flow, and whole-record flow with separate repeatability and model-sensitivity intervals.
4. Retained-charge availability.
5. Q–V traces for every voltage scan and all maximum-voltage conditions.
6. Percentage-of-breakdown fits, with transition points and legacy MAX kept out of the fitted models.

## What the reported quantities mean

The script detects the duty-burst frequency from each waveform activity envelope and detects the carrier separately. For each carrier-voltage half-cycle, it calculates the directed terminal-charge change from Channel D, subtracts the polarity-specific passive response learned from the 40%, 60%, and 75% breakdown scans, and applies a dielectric/cell correction only when a declared model is available. The unused 90% captures act as an out-of-sample passive-resolution check.

- **Positive/negative charge per peak half-cycle** is the p95 across the maximum-amplitude carrier half-cycle of the relevant polarity in each duty burst.
- **Peak rate** is p95 of `q/(e Δt)` across the same maximum-amplitude lobe selected once per duty burst. It is a half-cycle-average net charge-equivalent rate, not an instantaneous nanosecond microdischarge peak.
- **Whole-record rate** is the signed background-subtracted charge sum divided by elementary charge and the full record duration, including duty-off time.
- **Retained charge** is reported only when quiet, stable plateaus exist at both record edges. Positive/negative transport imbalance is never relabeled as retained surface charge.

“Negative carriers” means a net external-terminal negative electrical equivalent assigned to the plasma–liquid interface under the configured polarity mapping. By default, a negative pin-voltage lobe is assumed to mean negative-carrier delivery. Channel-D sign locking does not validate that physical assignment, and dielectric memory voltage can shift gas conduction relative to the source-voltage zero crossing. The waveform cannot separate electrons from negative ions or measure their gross counts independently. Likewise, the result is a total rate, not area-normalized flux, unless `--active-area-mm2` is supplied.

## Confidence and model status

`repeat_ci` resamples only the sequential MAX captures while holding calibration fixed. `analysis_ci` jointly resamples the 40%/60%/75% calibration captures within voltage level and the MAX captures in moving blocks; carrier half-cycles are not treated as independent replicates. These describe technical repeatability, not a population of independently rebuilt experiments. The broader model-sensitivity ranges additionally propagate the declared 0.1 µF monitor-capacitor tolerance, Channel-D gain tolerance, the shared effect of that scale on `Ccell`, the approximate full-base Pyrex geometry, and a bounded scalar-model choice from `C'` to `|C*|`.

The present scans support complex `Ccell` only where all passive consistency gates pass. A two-point 105%/115% slope is shown as a diagnostic but is deliberately not accepted as a publication-quality empirical `Cd` fit. For conductive-liquid conditions, a full-beaker-base Pyrex `Cd` scenario is therefore labeled exploratory. The raw terminal-excess result remains available separately from that model-dependent interpretation.

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
--no-pdf --no-plots
```

Run `python DBD_Surface_Charge_Analysis.py --help` for the complete list.
