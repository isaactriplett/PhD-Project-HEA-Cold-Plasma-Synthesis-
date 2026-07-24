# Lissajous v2: measured-frequency charge analysis

This pipeline extends the earlier archive-specific scripts without changing their
interfaces or outputs. It inventories the complete July 2026 tree, reads channel
roles and units from every PicoScope CSV header, measures carrier and burst
frequencies separately, quantifies Channel-D charge delivery, and creates the
contracted audit products used by both the preserved diagnostic figures and the
clear presentation figure set.

The central gate is deliberate: a folder named `4 kHz`, `10 kHz`, or `20 kHz`
does not receive an LF tag. `band_tag` is assigned from the measured Channel-A
electrical carrier. In the supplied operational records those labels are burst
rates and the measured carrier is generally 60–130 kHz, so their capacitances
are `carrier-transfer`, not `LF-geometric`.

## Run

From the repository root, with NumPy and Matplotlib installed:

```powershell
python -m lissajous.quantify --config config.yaml `
  --data-root "C:\Users\isaac\OneDrive\Documents\McGill\McGill Experiments\waveforms\July 2026" `
  --out outputs\lissajous_v2

# Preserve/regenerate the original eight-figure audit set.
python -m lissajous.figures --out outputs\lissajous_v2

# Regenerate the complete clear presentation set from saved outputs.
python -m lissajous.presentation_all `
  --out outputs\lissajous_v2 `
  --legacy-dir dbd_surface_charge_report
```

`config.yaml` is JSON-compatible YAML, so no YAML package is required. Change
the path and starred calibration/inventory entries there. A resolved copy is
written into the output folder.

`lissajous.presentation_all` does not rerun waveform quantification. It combines
the core and supplementary presentation renderers using the saved CSV/NPZ
products. Run `lissajous.figures` only when the original compact audit set is
also needed for traceability.

Matching top-level ZIP files are used as fast mirrors only when their logical
path and uncompressed size match the extracted source. Extracted-only files,
including the fourth 7_19 run and 7_21, are still included. Use
`--no-archives` to force direct reads.

Diagnostic subset options are available but are not part of the production run:

```powershell
python -m lissajous.quantify --path-regex "7_19/" --max-files 8
```

## Quantities and units

- `Q_nC = C_m × V_D`, after per-segment offset removal and recorded optional
  linear detrending.
- `Cline_pF = |Q̂(f0)/V̂(f0)|`; `Creal_pF`, `Cimag_pF`, and `phase_deg` retain
  the complex result. Numerically, nC/kV equals pF.
- `Clobe_pF` is the separate whole-record time-domain least-squares slope.
- `dQ_cycle_nC` is the Q-axis separation between rising and falling
  zero-voltage crossings of complete measured-carrier cycles.
- `dQ_gap_nC` applies only a same-band factor. The configured 1.15 factor is a
  carrier-transfer result and is never transferred to a genuine LF band.
- `U_cycle_signed_uJ` keeps loop orientation. A negative median in an active
  condition fails the orientation gate; the reported nonnegative
  `U_cycle_uJ`/`P_W` fields are then blank rather than silently absolutized.
- When bursts are present, `U_burst_uJ` is the cyclic shoelace area per complete
  burst period and `P_W = U_burst × f_burst`. If no valid nonnegative
  burst-period energy is available—because orientation is rejected or no
  complete burst-period window is present—power is withheld; the code does not
  silently fall back to `U_cycle × f0`. Continuous records use `U_cycle × f0`.
  `P_method` records the estimator or withholding reason per capture and
  condition.
- Positive/negative and gross/net charge rates are electrical charge
  equivalents. The waveforms cannot separate electrons from negative ions.
- The `*_peak_halfcycle_average_equivalent_flow_per_s` fields divide charge by
  a measured half-carrier period and by the elementary charge. They are useful
  carrier-equivalent flow rates, not instantaneous microdischarge peaks or
  area-normalized fluxes.
- Retained terminal charge is reported only when both record edges are quiet;
  it is suppressed when linear detrending was required.

Statistical intervals are capture-level medians and 2.5th–97.5th percentiles.
The correlated calibration contribution remains separate in `syst_frac`.

## QC behavior

- Every file remains in `manifest.csv`, including parse failures.
- Channel labels, units, role mapping, over-range counts, ADC code counts,
  offsets, drift, measured frequencies, duty, and exclusions are recorded.
- Fewer than 30 distinct codes is flagged as quantization-limited. Such a
  capture is not admitted as a geometric anchor.
- The 7_20 commanded-2-kV sets are flagged as surface-discharge-contaminated
  and excluded from displacement-only summaries.
- Condition-level charge polarity is locked from the capture votes. Pearson
  Channel C is independently sign-corrected against `dQ/dt`; it is not used to
  replace Channel-D charge.
- The spectral burst estimate is retained when a clipped threshold-envelope
  edge count locks to a harmonic; the failed edge cross-check is preserved.

## Output

`outputs/lissajous_v2/` contains the contracted CSV tables, the R3 consistency
gate, same-band `F(f)` table, onset table, machine-readable numbers pack,
`RESULTS.md`, and a deterministic figure-data cache.

- `figures/` and `captions/` contain the original eight-figure audit set. They
  are preserved for diagnostic traceability and are not the recommended slide
  sequence.
- `presentation_figures/` and `presentation_captions/` contain the clearer,
  one-message-per-page presentation set.
- `PRESENTATION_FIGURE_INDEX.md` gives the recommended main-story and appendix
  order, with a one-sentence takeaway and required caveat for every page.
- `presentation_core_manifest.csv` and `presentation_supplement_manifest.csv`
  map every clear-set figure ID to its PNG, PDF, and caption; the supplementary
  manifest also marks legacy/model-dependent pages explicitly.

The configured series-RLC element values carry a dagger throughout: the
original v1.2 fitting script/table is not present in this branch, so the code
evaluates the supplied order-of-magnitude model without presenting it as a new
fit.
