# DBD voltage-scan analysis

`Lissajous_Scan_Analysis.py` is the batch companion to the single-waveform
`Lissajous_Figures.py` script. It reads PicoScope CSV members directly from a
ZIP archive; it does not extract or alter the raw waveforms.

## Run

```powershell
python -m pip install -r requirements-lissajous.txt
python Lissajous_Scan_Analysis.py "C:\path\to\waveforms.zip" `
  --output-dir lissajous_voltage_scan_analysis `
  --files-per-level 16
```

Use `--files-per-level 0` to use all captures. Positive values select evenly
spaced capture indices over the full acquisition, not the first consecutive
files.

## Method and safeguards

- The 40%, 60%, and 75% records establish the passive response. The 90% and
  100% records are diagnostics because intermittent pre-breakdown activity can
  contaminate them.
- The passive carrier response is reported as
  `C* = C' - i C''`; the script rejects a scalar `Ccell` when the response has
  non-passive orientation or is not stable with voltage.
- The 105%, 115%, and old maximum-voltage extrema test the multi-amplitude
  `Qmax` relation. `Cd` is accepted only if the relation is monotonic, linear,
  and gives `Cd > Ccell` with a stable correction factor.
- The script checks whether the Pearson current is consistent with
  `I = Cref dVmonitor/dt`. A failed gain/phase check prevents the Pearson route
  from being labeled an absolute charge calibration.
- Negative charge is an electron-plus-negative-ion charge equivalent; the
  electrical data cannot separate those species.
- A p95 half-cycle-average rate is not the instantaneous nanosecond
  microdischarge peak.
- Retained charge is not reported unless both record edges are quiet. A
  positive/negative charge imbalance is not substituted for retained charge.

## Outputs

- `validated_results.csv`: shortest table of scientifically supported results.
- `scan_capacitance_results.csv`: both monitor and Pearson diagnostic fits.
- `max_voltage_charge_results.csv`: explicitly flagged provisional sensitivity
  ranges; failed routes remain visible for auditing.
- `file_level_qc_features.csv`: one row per selected waveform.
- `archive_manifest.csv`: every discovered raw CSV and its selection status.
- `analysis_audit.json`: machine-readable model decisions and limitations.
- `figures/*_scan_fit.png`: scan-level extrema plots.

The nominal 0.1 microfarad monitor-capacitor tolerance and probe-calibration
uncertainties scale the reported charge and must be supplied before publication.
