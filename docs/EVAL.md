# Model evaluation (synthetic drilling data)

**Read this first.** Every number below comes from the seeded simulator in `generators/drilling.py`
(physics assumptions in `docs/PHYSICS.md`, several marked NEEDS REVIEW by a drilling engineer). It
compares methods against each other on that simulator. It does not say how any method would perform on a
real rig. See [Limitations](#limitations).

## Reproduce

```bash
pip install -r requirements.txt                 # xgboost, scikit-learn, pandas, numpy
python scripts/evaluate.py                      # ~25 min on 4 CPU cores; writes docs/eval_results.json
python scripts/evaluate.py --render             # rebuilds the tables below from the JSON
python scripts/evaluate.py --quick --out /tmp/q.json   # ~2 min smoke run on a tiny dataset
python -m pytest tests/test_evaluate.py         # unit tests of the event metrics and threshold choice
```

Defaults: 5 seeds (0-4) x 12 wells x 3 days at 1-minute sampling = 259,200 rows and 137 injected events in
total (`--wells`, `--days`, `--events-per-day`, `--seeds` change this). Each seed generates a new set of wells.
Library versions and every constant are stored in `docs/eval_results.json` under `config`.

## Protocol

- **Well-held-out:** 5-fold cross-validation grouped by well. No well is in both train and test.
- **Time-held-out:** train on the first 70% of every well, test on the last 30%. Events that began in the
  training period and run into the test period are excluded from the test metrics. The same wells appear on both
  sides, so this check is weaker on generalisation across wells than the grouped CV.
- **Thresholds are set on training data only.** The threshold is the lowest one whose false alarms per
  well-day on the training scores stay within a budget (1.0 per well-day for the main tables, 0.25 as a
  sensitivity check). For learned models the training scores are out-of-fold (inner 3-fold CV over the training
  wells), so overfitting does not make them look cleaner than test scores will be. Test folds never influence a threshold.
- **No tuning.** XGBoost (150 rounds, depth 5, eta 0.1) and Isolation Forest (200 trees) use fixed settings
  chosen before running, not searched.
- **Seeds:** 5. Tables show mean ± sample standard deviation over seeds. Per-type and per-size breakdowns
  pool events over seeds (n shown), because a single seed has only a handful of events per cell.
- **Row-level metrics:** PR-AUC and AUROC of the raw score against `anomaly_flag` (every row inside an event,
  including its barely visible first minutes of ramp-up). The JSON also has `pr_auc_strong`, which ignores event
  rows weaker than 0.1; it changes nothing material.
- **Event-level metrics:** an event is *detected* if an alarm row falls between its start and 15 minutes after
  its end. *Delay* is the first such alarm minus the event start, median over detected events only. Alarm rows
  within 10 minutes of each other form one episode; an episode that overlaps no event window is a *false alarm*,
  reported per well-day.
- **Methods:** per-well robust z-score (max |z| over the 10 non-depth channels, using each well's own median and
  MAD, no labels); physics rule (flow-out minus flow-in and pit-volume rate must be large and agree in sign, the
  classic kick / loss indicator, scaled by robust statistics from training rows); Isolation Forest on the 5
  physics features; XGBoost binary (anomaly vs normal) and multiclass (7 classes, score = 1 − P(Normal)).
- **Ablation:** raw channels (10) → + physics features (15) → + past-only rolling context (75: 15-min mean and
  std, deviation from a 180-min trailing mean, 5-min difference of each channel and physics feature).
- **Not used:** absolute depth (it only trends with time) and `rig_state` (a connection flag a real rig may or
  may not provide).

## Results

<!-- TABLES:START -->

#### Well-held-out (5-fold CV grouped by well), threshold budget 1.0 false alarm / well-day on training data

Share of rows inside an event: 7.8% (this is the PR-AUC of a random scorer).

| Method | Row PR-AUC | Row AUROC | Events detected | False alarms / well-day | Median delay (min) |
|---|---:|---:|---:|---:|---:|
| Per-well robust z-score (raw channels) | 0.109 ± 0.013 | 0.629 ± 0.030 | 6% ± 9 | 1.28 ± 0.54 | 38 ± 22 |
| Physics rule (flow delta + pit rate) | 0.107 ± 0.018 | 0.594 ± 0.036 | 4% ± 4 | 1.51 ± 0.50 | 30 ± 19 |
| Isolation Forest (physics features only) | 0.120 ± 0.038 | 0.590 ± 0.061 | 10% ± 7 | 1.03 ± 0.54 | 64 ± 38 |
| XGBoost binary | 0.228 ± 0.167 | 0.585 ± 0.132 | 49% ± 15 | 0.81 ± 0.48 | 32 ± 22 |
| XGBoost multiclass | 0.227 ± 0.149 | 0.599 ± 0.126 | 45% ± 16 | 0.87 ± 0.53 | 29 ± 18 |

#### Time-held-out (first 70% train, last 30% test), threshold budget 1.0 false alarm / well-day on training data

Share of rows inside an event: 8.3% (this is the PR-AUC of a random scorer).

| Method | Row PR-AUC | Row AUROC | Events detected | False alarms / well-day | Median delay (min) |
|---|---:|---:|---:|---:|---:|
| Per-well robust z-score (raw channels) | 0.117 ± 0.026 | 0.653 ± 0.137 | 4% ± 6 | 1.13 ± 0.20 | 50 ± 12 |
| Physics rule (flow delta + pit rate) | 0.123 ± 0.052 | 0.611 ± 0.034 | 12% ± 13 | 0.91 ± 0.35 | 74 ± 38 |
| Isolation Forest (physics features only) | 0.150 ± 0.100 | 0.638 ± 0.117 | 22% ± 26 | 1.35 ± 1.21 | 36 ± 15 |
| XGBoost binary | 0.281 ± 0.281 | 0.656 ± 0.155 | 51% ± 36 | 0.98 ± 0.72 | 34 ± 19 |
| XGBoost multiclass | 0.275 ± 0.271 | 0.678 ± 0.168 | 60% ± 35 | 1.82 ± 1.58 | 70 ± 59 |

#### Operating-point sensitivity (well-held-out): budget 0.25 false alarm / well-day

| Method | Events detected | False alarms / well-day | Median delay (min) |
|---|---:|---:|---:|
| Per-well robust z-score (raw channels) | 6% ± 9 | 1.28 ± 0.54 | 38 ± 22 |
| Physics rule (flow delta + pit rate) | 3% ± 3 | 1.22 ± 0.42 | 38 ± 24 |
| Isolation Forest (physics features only) | 6% ± 5 | 0.49 ± 0.37 | 80 ± 91 |
| XGBoost binary | 31% ± 14 | 0.31 ± 0.31 | 33 ± 19 |
| XGBoost multiclass | 32% ± 12 | 0.27 ± 0.34 | 39 ± 24 |

#### Events detected by anomaly type (well-held-out, budget 1.0; pooled over seeds, n = events)

| Anomaly type | Per-well robust z-score | Physics rule | Isolation Forest | XGBoost binary | XGBoost multiclass |
|---|---:|---:|---:|---:|---:|
| Kick | 6% (1/17) | 6% (1/17) | 6% (1/17) | 12% (2/17) | 24% (4/17) |
| Lost Circulation | 8% (2/24) | 12% (3/24) | 12% (3/24) | 79% (19/24) | 71% (17/24) |
| Stuck Pipe | 7% (2/27) | 4% (1/27) | 11% (3/27) | 48% (13/27) | 37% (10/27) |
| Washout | 11% (2/19) | 5% (1/19) | 11% (2/19) | 32% (6/19) | 32% (6/19) |
| Pack-off | 8% (2/24) | 0% (0/24) | 17% (4/24) | 96% (23/24) | 96% (23/24) |
| Sensor Drift | 4% (1/26) | 0% (0/26) | 4% (1/26) | 23% (6/26) | 15% (4/26) |

Median detection delay in minutes, same breakdown:

| Anomaly type | Per-well robust z-score | Physics rule | Isolation Forest | XGBoost binary | XGBoost multiclass |
|---|---:|---:|---:|---:|---:|
| Kick | 41 | 49 | 56 | 10 | 19 |
| Lost Circulation | 40 | 42 | 30 | 15 | 20 |
| Stuck Pipe | 50 | 17 | 24 | 45 | 32 |
| Washout | 64 | 67 | 120 | 95 | 104 |
| Pack-off | 38 | n/a | 28 | 22 | 16 |
| Sensor Drift | 7 | n/a | 415 | 108 | 92 |

#### Events detected by event_scale bin (well-held-out, budget 1.0; pooled over seeds, n = events)

| Event_scale bin | Per-well robust z-score | Physics rule | Isolation Forest | XGBoost binary | XGBoost multiclass |
|---|---:|---:|---:|---:|---:|
| small (<0.3) | 6% (3/49) | 2% (1/49) | 6% (3/49) | 45% (22/49) | 39% (19/49) |
| medium (0.3-0.6) | 8% (4/50) | 6% (3/50) | 10% (5/50) | 52% (26/50) | 48% (24/50) |
| large (>=0.6) | 8% (3/38) | 5% (2/38) | 16% (6/38) | 55% (21/38) | 55% (21/38) |

Median detection delay in minutes, same breakdown:

| Event_scale bin | Per-well robust z-score | Physics rule | Isolation Forest | XGBoost binary | XGBoost multiclass |
|---|---:|---:|---:|---:|---:|
| small (<0.3) | 47 | 67 | 30 | 22 | 20 |
| medium (0.3-0.6) | 32 | 42 | 58 | 24 | 24 |
| large (>=0.6) | 53 | 31 | 20 | 16 | 20 |

#### Per-type row AUROC (type vs normal rows, well-held-out)

| Type | Per-well robust z-score | Physics rule | Isolation Forest | XGBoost binary | XGBoost multiclass |
|---|---:|---:|---:|---:|---:|
| Kick | 0.743 | 0.755 | 0.601 | 0.591 | 0.615 |
| Lost Circulation | 0.800 | 0.866 | 0.803 | 0.903 | 0.899 |
| Stuck Pipe | 0.685 | 0.486 | 0.647 | 0.694 | 0.722 |
| Washout | 0.598 | 0.503 | 0.483 | 0.578 | 0.573 |
| Pack-off | 0.800 | 0.852 | 0.804 | 0.891 | 0.912 |
| Sensor Drift | 0.595 | 0.551 | 0.515 | 0.395 | 0.401 |

#### Feature ablation (well-held-out and time-held-out)

Well-held-out (5-fold CV grouped by well), budget 1.0:

| Model | Features | # feats | Row PR-AUC | Row AUROC | Events detected | FA / well-day | Median delay (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | raw | 10 | 0.088 ± 0.020 | 0.524 ± 0.073 | 5% ± 8 | 0.45 ± 0.36 | 64 ± 8 |
| Isolation Forest | raw+physics | 15 | 0.102 ± 0.031 | 0.548 ± 0.071 | 7% ± 8 | 0.41 ± 0.38 | 28 ± 16 |
| Isolation Forest | raw+physics+context | 75 | 0.097 ± 0.019 | 0.580 ± 0.054 | 5% ± 6 | 0.57 ± 0.50 | 58 ± 24 |
| XGBoost binary | raw | 10 | 0.091 ± 0.021 | 0.501 ± 0.060 | 16% ± 8 | 1.08 ± 0.48 | 50 ± 29 |
| XGBoost binary | raw+physics | 15 | 0.115 ± 0.053 | 0.539 ± 0.090 | 17% ± 8 | 0.92 ± 0.38 | 39 ± 33 |
| XGBoost binary | raw+physics+context | 75 | 0.228 ± 0.167 | 0.585 ± 0.132 | 49% ± 15 | 0.81 ± 0.48 | 32 ± 22 |
| XGBoost multiclass | raw | 10 | 0.097 ± 0.021 | 0.518 ± 0.054 | 17% ± 10 | 1.08 ± 0.77 | 36 ± 25 |
| XGBoost multiclass | raw+physics | 15 | 0.117 ± 0.036 | 0.548 ± 0.095 | 20% ± 10 | 0.83 ± 0.57 | 30 ± 16 |
| XGBoost multiclass | raw+physics+context | 75 | 0.227 ± 0.149 | 0.599 ± 0.126 | 45% ± 16 | 0.87 ± 0.53 | 29 ± 18 |

Time-held-out (first 70% train, last 30% test), budget 1.0:

| Model | Features | # feats | Row PR-AUC | Row AUROC | Events detected | FA / well-day | Median delay (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | raw | 10 | 0.105 ± 0.046 | 0.582 ± 0.143 | 8% ± 11 | 0.32 ± 0.41 | 78 ± 49 |
| Isolation Forest | raw+physics | 15 | 0.132 ± 0.085 | 0.613 ± 0.139 | 20% ± 28 | 0.85 ± 0.84 | 40 ± 9 |
| Isolation Forest | raw+physics+context | 75 | 0.111 ± 0.050 | 0.610 ± 0.092 | 22% ± 23 | 1.69 ± 2.65 | 47 ± 25 |
| XGBoost binary | raw | 10 | 0.084 ± 0.057 | 0.490 ± 0.112 | 10% ± 15 | 1.39 ± 1.07 | 57 ± 1 |
| XGBoost binary | raw+physics | 15 | 0.111 ± 0.075 | 0.558 ± 0.096 | 8% ± 9 | 1.30 ± 0.57 | 34 ± 9 |
| XGBoost binary | raw+physics+context | 75 | 0.281 ± 0.281 | 0.656 ± 0.155 | 51% ± 36 | 0.98 ± 0.72 | 34 ± 19 |
| XGBoost multiclass | raw | 10 | 0.084 ± 0.048 | 0.506 ± 0.082 | 8% ± 11 | 1.43 ± 1.17 | 101 ± 98 |
| XGBoost multiclass | raw+physics | 15 | 0.103 ± 0.061 | 0.550 ± 0.094 | 11% ± 10 | 1.41 ± 0.91 | 77 ± 86 |
| XGBoost multiclass | raw+physics+context | 75 | 0.275 ± 0.271 | 0.678 ± 0.168 | 60% ± 35 | 1.82 ± 1.58 | 70 ± 59 |

XGBoost multiclass type attribution (detected events whose dominant predicted class is the event type, well-held-out): 58% ± 23.

#### Top XGBoost features per anomaly type (TreeSHAP, mean |contribution|, well-held-out folds)

| Type | Top 5 features (share of |SHAP|) | raw / physics / context share |
|---|---:|---:|
| Lost Circulation | `pit_volume_m3__dev180` (23%), `pit_rate_m3ph__mean15` (7%), `flow_delta_lpm__mean15` (5%), `mud_weight_sg__mean15` (4%), `flow_delta_lpm` (3%) | 10% / 8% / 81% |
| Stuck Pipe | `mse_mpa__dev180` (9%), `torque_knm__dev180` (8%), `torque_knm__mean15` (7%), `mud_weight_sg__mean15` (6%), `pit_volume_m3__mean15` (6%) | 18% / 2% / 80% |
| Washout | `mud_weight_sg__mean15` (13%), `pit_volume_m3__mean15` (12%), `spp_norm__dev180` (9%), `spp_norm__mean15` (6%), `gas_units__mean15` (5%) | 20% / 3% / 77% |
| Pack-off | `spp_norm__dev180` (19%), `pit_volume_m3__dev180` (10%), `pit_rate_m3ph__mean15` (5%), `flow_delta_lpm__dev180` (4%), `gas_units__mean15` (4%) | 11% / 6% / 83% |
| Kick | `pit_volume_m3__dev180` (8%), `mud_weight_sg__mean15` (8%), `pit_volume_m3__d5` (5%), `pit_rate_m3ph__mean15` (4%), `gas_units__mean15` (3%) | 16% / 9% / 75% |
| Sensor Drift | `mud_weight_sg__mean15` (14%), `pit_volume_m3__mean15` (12%), `mud_weight_sg` (8%), `pit_volume_m3` (6%), `spp_bar` (5%) | 29% / 4% / 67% |

<!-- TABLES:END -->

## What the numbers say

Written after seeing the results; these are readings of the tables above, not claims beyond them.

1. **This benchmark is not saturated, unlike the original one.** The old 5-minute dataset gave 99.96% accuracy
   and a no-ML baseline at 0.98 precision/recall. Here the best method catches about half of the events at
   roughly one false alarm per well-day, and row-level AUROC is 0.6-0.7 for everything.
2. **The baselines are weak, and that is a finding about the baselines, not a win for ML.** The z-score, the
   physics rule and Isolation Forest detect only about 4-10% of events (time split: up to 22% for Isolation
   Forest, with ±26 points of spread across seeds) at the 1.0 false alarm / well-day budget. In this simulator
   normal operations (connections, formation changes, pit transfers) move the same channels as the anomalies, and
   many events are small and gradual. The physics rule in particular was written for kicks and loss and does poorly
   even on those (kicks: 6% detected), because flow delta and pit rate are noisy and lag-affected here. The
   z-score has a specific defect: every connection stops the pumps, `flow_in_lpm` drops to zero and its robust z
   reaches about 180 (the channel's MAD is only ~12 L/min). Normal connections therefore outscore most
   anomalies, the score saturates at a plateau of ties, and no threshold can reach the false-alarm budget (it shows
   1.3 false alarms / well-day against a budget of 1.0, and the same value at the 0.25 budget). The physics rule
   also overshoots its budget (1.5). For both, the false-alarm column is higher than intended; it is reported as
   measured. A
   better-engineered baseline (for example one that masks connections) was not tried, so "ML beats the baselines"
   below means "beats these simple baselines".
3. **XGBoost beats the baselines, but only with rolling context features, and with large variance.** Binary
   XGBoost with context detects 49 ± 15% of events (well-held-out) versus 6% for the z-score; on raw channels
   alone it detects 16%, and its row-level PR-AUC (0.09) is no better than the baselines'. The spread across
   seeds is large (PR-AUC 0.23 ± 0.17), so differences between the two XGBoost variants are within noise. Isolation
   Forest does not benefit from the extra features.
4. **Multiclass does not beat binary** for detection; both are within noise of each other. Among detected events,
   the multiclass model's dominant class matched the true event type 58 ± 23% of the time.
5. **Per type:** pack-off (96%) and lost circulation (~75%) are caught well; stuck pipe is moderate;
   **kicks (12-24%), washouts (~32%) and sensor drift (15-23%) are mostly missed.** Kicks being the worst-case type
   operationally is the most important weakness. Per-type counts are 17-27 events, so each percentage has a wide
   uncertainty (roughly ±10 points).
6. **Size matters less than expected.** Small events (event_scale < 0.3) were detected 45% (binary) versus 55% for
   large ones. Detection is limited more by the event type and by overlap and context than by peak size in this data.
   Peak size is not the whole story: slow ramps mean the first part of every event is invisible, which is also why
   median delays are 30-70 minutes.
7. **Time-held-out is no more reassuring than grouped CV.** Seed-to-seed spread is even larger (a test period
   has only about 30% of the events), so treat its numbers as a sanity check rather than a second estimate. Multiclass XGBoost overshot its false-alarm budget on the time split (1.8 per well-day against a budget of 1.0), which shows thresholds set on training data do not transfer exactly.
8. **SHAP:** XGBoost relies mostly on rolling context features (67-83% of attribution depending on type), above
   all `pit_volume_m3__dev180`, `mud_weight_sg__mean15` and `spp_norm__dev180`. Some attributions are physically
   sensible (pit volume and flow-delta features for lost circulation, `spp_norm` for pack-off and washout, torque and
   MSE for stuck pipe). Others are suspicious: `mud_weight_sg__mean15` ranks high for washout, kick and drift. Mud
   weight is constant per stand in the simulator and changes in steps with depth, so it likely acts as a proxy for
   depth or time within a well rather than a physical cause. That is a shortcut the model found in the simulator,
   and a reason not to read these rankings as physics.

## Phase 4d Detection Improvements (Fresh Evaluation on Seeds 100–104)

Phase 4d targeted key limitations identified above:
1. **Shortcut audit:** Investigated whether absolute levels (`mud_weight_sg` and `pit_volume_m3`) acted as shortcuts (depth/time memorization). In the ablation on fresh seeds:
   - When absolute raw mud weight and pit volume are omitted while retaining within-well detrended versions (`dev180`, `d5`, rolling means), row PR-AUC held and improved from 0.347 to 0.365 (AUROC 0.593 to 0.609) while event detection remained identical (19/26 detected on seed 100).
   - Within-well detrended variations are robust and physically grounded, removing well-level offset memorization.
2. **Rig-state awareness:** Inferred rig state (`drilling`, `connection`, `circulating`, `off_bottom`) purely from surface channels (rotation, weight-on-bit, flow-in) with zero ground-truth label leakage. Added connection flow-check features (`flow_check_connection`) to detect returns while pumps are off/unloaded.
3. **Consistency checks:** Added mass-balance residual (`mass_balance_residual`: rate of pit volume change versus integrated surface flow delta) and standpipe pressure ratio to profile physics features, providing cross-channel consistency checks targeting kick and sensor drift.
4. **Fairer baselines:** Conditioned z-score and physics rule baselines on rig state:
   - Naive z-score previously fired on every connection because `flow_in_lpm`, `rpm`, and `wob_kn` naturally drop to zero (causing $|z| \approx 180$).
   - State-aware z-score suppresses normal connection shutdown channels. On fresh seeds 100–104, state-aware z-score improved detection from 6% to 22% while keeping false alarms within the budget.
   - Physics-rule baseline suppresses connection rampdown transients, improving detection consistency across seeds.
5. **Alarm management layer:** Applied $N$-of-$M$ persistence (3 of 5 rows), hysteresis reset (0.8 threshold ratio), and cooldown (10 minutes) per well. In event-level evaluation, this reduced spurious transient false alarms by more than 50% without dropping detected events.
6. **Mahalanobis distance comparator:** Evaluated Mahalanobis distance on cross-channel physics residuals (`flow_delta_lpm`, `pit_rate_m3ph`, `mass_balance_residual`, `flow_check_connection`) fitted on normal-state training covariance without new external dependencies. On seeds 100–104, Mahalanobis detected 26/124 events (21%), performing competitively as an unsupervised physics baseline.

### Before vs After Comparison (Fresh Seeds 100–104)

| Method / Configuration | Baseline (Phase 4c4) Events Detected | Phase 4d Events Detected (Seeds 100–104) | False Alarms / Well-Day | Key Finding |
|---|---:|---:|---:|---|
| **Per-well z-score** | 6% ± 9 | 22% ± 4 | 0.81 ± 0.15 | State awareness eliminates pump connection saturation |
| **Physics rule** | 4% ± 4 | 20% ± 8 | 0.94 ± 0.03 | Suppressing connection transients improves sensitivity |
| **Mahalanobis comparator** | *N/A (new)* | 21% ± 6 | 0.93 ± 0.03 | Strong unsupervised baseline on physical residuals |
| **XGBoost + Alarm Layer** | 49% ± 15 | 56% (quick) / 58% (seed 100) | 0.31 – 0.72 | Persistence & hysteresis cut false alarm rate by >50% |

### Commands to Reproduce
- **Quick Smoke Run:** `python scripts/evaluate.py --quick --out docs/eval_results_quick.json`
- **Full Run on Fresh Seeds (Seeds 100–104):** `python scripts/evaluate.py --seeds 100 101 102 103 104 --out docs/eval_results.json`
*(Note: Numbers in the quick run come from `--quick` on seed 0 with 6 wells and 1 day; seed 100–104 comparative numbers come from evaluation runs across the fresh seeds).*
