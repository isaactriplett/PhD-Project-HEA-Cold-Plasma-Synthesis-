# DBD surface-charge reporting analysis

Analysis version: `2.0-surface-charge-power`

## Definitions

The nominal monitor charge is `Qm(t) = Cref Vmonitor(t)`, with
`Cref = 0.1 µF`. Channel-D polarity is inferred
once from the 40/60/75 %-breakdown ensemble for each independently scanned
condition and then locked for every percentage and legacy MAX capture.
This sign lock orients the electrical monitor; it does not establish which
carrier species reaches the liquid. The default reporting assignment treats a
negative pin-voltage lobe as negative-carrier delivery, and every table/figure
records that as an explicit, unvalidated physical assumption.

The duty-burst frequency is measured from the waveform activity envelope for
each capture. Folder frequency is used only if envelope detection falls back to
the carrier or fails; a >15 % mismatch fails passive-model validation.

Adjacent interpolated carrier-voltage zero crossings define a half-cycle. For
voltage polarity `s = ±1`, the directed terminal charge is

`D_h = s [Qm(t1) - Qm(t0)]`.

Separate passive functions for positive and negative voltage lobes are fitted
from 40/60/75 % data. The operational terminal excess is

`X_h = D_h - B_s(A_h)`.

The 90 % captures are excluded from that fit and used as an out-of-sample
passive null. Peak metrics must exceed the larger of the training-residual
limit and the 90 %-holdout p95 at the lower endpoint of the joint 95 % analysis
interval. Same-frequency whole-record metrics are compared with the same
90 %-holdout statistic. Transferred-frequency whole-record results have no
same-frequency holdout and are labeled accordingly.

When a valid dielectric capacitance is available, the classical model gives

`q_surface,h = [Cd/(Cd-Ccell)] X_h`.

Geometry-derived `Cd` results are explicitly exploratory. The active 105/115 %
secant is diagnostic because only two independent active amplitudes are
available; legacy MAX is out-of-sample and is never used to fit `Cd`.

## Requested outputs

- **Charge per peak half-cycle:** p95 across the maximum-amplitude carrier lobe
  of each polarity in each duty burst.
- **Peak rate:** p95 of half-cycle-average `q/(e Δt)` across the same
  maximum-amplitude lobe selected once per duty burst. It is not an
  instantaneous nanosecond particle flux.
- **Whole-record rate:** signed sum of background-subtracted lobe charge divided
  by elementary charge and full record duration, including duty-off time.
- **Retained charge:** only measured when stable quiet plateaus exist at both
  record edges. Polarity imbalance is reported separately and never relabeled
  retained surface charge.

Negative rate means a **net external-terminal electrical equivalent** assigned
to electrons plus negative ions under the configured pin-polarity mapping. It
is not a species-resolved gross particle count, and memory voltage can shift
actual gas conduction relative to a source-voltage zero crossing.
Area-normalized flux is blank unless an active area is supplied.

## Uncertainty

The independent sampling unit is a waveform capture, not a carrier half-cycle.
`repeat_ci` is a conditional MAX-capture repeatability interval with the passive
calibration fixed. `analysis_ci` jointly resamples 40/60/75 % calibration
captures within level and legacy MAX captures using a
200-replicate moving-block bootstrap with block length
4. These are technical-repeat intervals; the 64
sequential captures are not independent biological or experimental repeats.

Broad model-sensitivity intervals additionally sample the declared monitor
scale and approximate full-base Pyrex geometry, propagate the common Channel-D
scale through both terminal charge and Ccell in `Cd/(Cd-Ccell)`, and span the
scalar-cell choice from `C'` to `|C*|`. These are bounded scenario ranges, not
frequentist confidence intervals, and they still do not cover unknown active
area or every possible circuit-model error. Invalid correction-factor draws
are counted and the result is not reportable if more than 20 % are unphysical.

The nominal monitor-capacitance scale sensitivity is ±
10.0 %, and Channel-D gain
sensitivity is ±3.0 %.
