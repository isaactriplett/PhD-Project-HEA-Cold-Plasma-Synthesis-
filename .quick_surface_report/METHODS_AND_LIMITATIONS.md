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

The passive response is estimated by projecting every usable sample in each
40/60/75 %-breakdown waveform onto in-phase and quadrature carrier bases. The
least-squares model is

`Qm(t) = C' V_I(t) + C'' V_Q(t) + offset + linear drift`.

This whole-waveform complex-C* regression replaces the deprecated lobe-amplitude
ratio fit, which could lock onto one ADC-code ratio. A passive capture with less
than eight peak-to-peak Channel-D codes remains explicitly quantization-limited;
the regression cannot create information absent from the acquisition.

Adjacent interpolated carrier-voltage zero crossings define a half-cycle. For
voltage polarity `s = ±1`, the directed terminal charge is

`D_h = s [Qm(t1) - Qm(t0)]`.

The fitted in-phase and quadrature carrier values are interpolated at both lobe
endpoints. Their exact endpoint changes give the complex passive prediction
`B_h(C', C'')`; the operational terminal excess is

`X_h = D_h - B_h(C', C'')`.

The 90 % captures are excluded from that fit and used as an out-of-sample
passive null. Peak metrics must exceed the larger of the training-residual
limit and the 90 %-holdout p95 at the lower endpoint of the joint 95 % analysis
interval. Same-frequency whole-record metrics are compared with the same
90 %-holdout statistic. Transferred-frequency whole-record results have no
same-frequency holdout and are labeled accordingly.

When a valid dielectric capacitance is available, the classical model gives

`q_surface,h = [Cd/(Cd-Ccell)] X_h`.

The effective active-branch `Cd` fit uses the capture medians at 100, 105, and
115 % breakdown, with an intercept. Captures are resampled within commanded
level. Use requires unclipped records, at least
4 clean captures per level, resolved activity
at 100 %, monotonic voltage and charge, `R² ≥ 0.98`,
consistent pairwise slopes, and at least 95 % physical bootstrap draws with
`Cd > 1.05 Ccell`. A passing three-level result is labeled **provisional**, not
fully validated; a denser active scan or direct dielectric measurement remains
preferable. Legacy MAX is out-of-sample and is never fitted. The full-base
Pyrex geometry result is retained as a separate model scenario rather than
being blended into the statistical confidence interval.

## Requested outputs

- **Charge per peak half-cycle:** within each capture and polarity, select the
  maximum-amplitude carrier lobe once per duty burst and calculate p95; report
  the median of those capture-level p95 values. Lobes from 64 captures are not
  pooled into one p95.
- **Peak rate:** p95 of half-cycle-average `q/(e Δt)` across the same
  maximum-amplitude lobe selected once per duty burst. It is not an
  instantaneous nanosecond particle flux.
- **Whole-record rate:** signed sum of background-subtracted lobe charge divided
  by elementary charge and full record duration, including duty-off time.
- **Retained charge:** only measured when stable quiet plateaus exist at both
  record edges. Polarity imbalance is reported separately and never relabeled
  retained surface charge.
- **Raw Q–V energy and apparent power:** for each complete duty-burst period,
  `E = 0.5 |Σ V_i (Q_{i+1} - Q_{i-1})|`, with cyclic indices, kV, and nC,
  so `E` is in µJ. A capture is represented by its median period energy and
  `P = f_burst E`; condition intervals resample captures. This is total raw
  Lissajous reactor input loss (plasma + dielectric + liquid + phase-skew), not
  plasma-only power.
- **Detected duty-on fraction:** time fraction above the same envelope threshold
  `P10 + 0.30(P90-P10)` within the same duty-period windows. The detector channel
  is audited because current- and voltage-envelope fractions are not identical
  physical quantities.
- **Dose clock:** ideal minutes to one negative-charge equivalent per ion are
  `c_mM V_mL N_A 10^-6 / (60 R_-)`, using a default volume of
  2.5 mL and concentration of
  5 mM. It assumes 100 % delivery/utilization,
  includes electrons plus negative ions, and is not a chemical conversion time.
  BMIM curves are hypothetical rate-transfer comparisons because BMIM nitrate
  contains no metal; Mn²⁺ reduction requires at least two equivalents.

Negative rate means a **net external-terminal electrical equivalent** assigned
to electrons plus negative ions under the configured pin-polarity mapping. It
is not a species-resolved gross particle count, and memory voltage can shift
actual gas conduction relative to a source-voltage zero crossing.
Area-normalized flux is blank unless an active area is supplied.

## Uncertainty

The independent sampling unit is a waveform capture, not a carrier half-cycle.
`repeat_ci` is a conditional MAX-capture repeatability interval with the passive
calibration fixed. `analysis_ci` jointly resamples 40/60/75 % complex-C*
calibration captures within level, 100/105/115 % active-Cd captures when that
model passes, and legacy MAX captures using a
200-replicate moving-block bootstrap with block length
4. These are technical-repeat intervals; the 64
sequential captures are not independent biological or experimental repeats.

Broad model-sensitivity intervals additionally sample the declared monitor
scale and keep the measured active-Cd and approximate full-base Pyrex geometry
as separate sampled scenarios. The common Channel-D scale is propagated through
both terminal charge and Ccell in `Cd/(Cd-Ccell)`, and the geometry-only branch
spans the scalar-cell choice from `C'` to `|C*|`. These are bounded scenario ranges, not
frequentist confidence intervals, and they still do not cover unknown active
area or every possible circuit-model error. Invalid correction-factor draws
are counted and the result is not reportable if more than 20 % are unphysical.

The nominal monitor-capacitance scale sensitivity is ±
10.0 %, and Channel-D gain
sensitivity is ±3.0 %.
