# Calibrated overlay filter benchmark

Selected on TUNE: **one-euro_fc=1.2_beta=0.004** — passed accuracy/response guards but reached only 17.7% TUNE jitter reduction, short of the proposed 50% goal.
T1 was evaluated only after selection.

Selection rule: lowest TUNE jitter P95 subject to <=5% median/P95 error change and <=100ms WORST-CASE step across 8 representative steps.

## Responsiveness (filter only, synthetic)

Headline 40%-width step: **33.33 ms** (adaptive candidate) vs **966.67 ms** worst case for the high-stability preset (fc=0.4, beta=0).

Adaptive worst case across 8 steps: **100.00 ms**; median **66.67 ms**.

| step case | adaptive (ms) | high-stability (ms) |
|---|---:|---:|
| x_small_+0.10 | 66.67 | 966.67 |
| x_small_-0.10 | 66.67 | 966.67 |
| x_mid_+0.20 | 66.67 | 966.67 |
| x_mid_-0.20 | 66.67 | 966.67 |
| x_large_+0.40 | 33.33 | 966.67 |
| x_large_-0.40 | 33.33 | 966.67 |
| y_mid_+0.20 | 100.00 | 966.67 |
| y_mid_-0.20 | 100.00 | 966.67 |

These are the filter's own settling times at 30 fps. They are NOT camera-to-display latency and do not evidence the specification's 50 ms end-to-end requirement.

## TUNE (used for selection)

| TUNE metric | unfiltered | selected filter |
|---|---:|---:|
| jitter radius P95 (px) | 116.33 | 95.73 |
| median error (px) | 184.29 | 182.11 |
| P95 error (px) | 609.36 | 605.38 |

## T1 (three arms, identical samples)

| T1 metric | unfiltered | high-stability | adaptive |
|---|---:|---:|---:|
| jitter P95 pooled (px, outlier-led) | 380.81 | 159.13 | 350.53 |
| jitter median-of-fixation-P95 (px) | 125.31 | 72.30 | 115.54 |
| jitter worst fixation P95 (px) | 698.54 | 253.54 | 699.04 |
| jitter radius median (px) | 58.04 | 26.88 | 39.40 |
| fixations counted | 10.00 | 10.00 | 10.00 |
| frame jump P95 (px) | 147.79 | 25.50 | 82.89 |
| mean error (px) | 271.82 | 268.97 | 268.29 |
| median error (px) | 209.10 | 196.14 | 196.15 |
| P90 error (px) | 570.41 | 582.83 | 567.32 |
| P95 error (px) | 747.82 | 639.36 | 715.09 |
| valid coverage (%) | 100.00 | 100.00 | 100.00 |
| recorded transition to 90% median (ms) | 1470.98 | 1469.64 | 1469.64 |
| recorded transition to 90% P90 (ms) | 1582.79 | 1477.82 | 1554.86 |
| transitions measured | 8.00 | 9.00 | 9.00 |
| transitions never reached | 0.00 | 0.00 | 0.00 |
| filter CPU mean (ms) | 0.0010 | 0.0019 | 0.0019 |

## Calibration support (filter-independent)

Maximum RBF kernel activation of the recorded features against the calibration's support vectors. Near 0 the SVR returns its constant bias, so the point stops following the eye rather than merely getting noisier. No filter can repair those samples.

| split | median activation | min | % below 0.01 |
|---|---:|---:|---:|
| tune | 0.909 | 0.663 | 0.00 |
| test | 0.898 | 0.000 | 2.38 |

## Limits

- synthetic step time measures only the filter, not camera-to-display latency
- recorded transition time includes operator reaction and model error
- stationary-target jitter includes residual eye settling and drift during collection
- round2/TUNE and round6/T1 have both already informed earlier decisions; neither is an untouched holdout
- hardware confirmation is still required
