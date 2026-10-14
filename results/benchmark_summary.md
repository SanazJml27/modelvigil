## Overall (all scenarios x seeds)

| Method | False-alarm rate (stable + benign) | Detection rate (harmful) | Median delay, weeks | Root-cause accuracy | Correct action |
|---|---|---|---|---|---|
| ModelVigil | 0% | 100% | 3.0 | 100% | 100% |
| naive_drift | 50% | 60% | 0.0 | — | — |
| perf_only | 0% | 80% | 4.0 | — | — |

## ModelVigil by scenario

| Scenario | Truth | Detected | Median delay | Root cause correct | Action correct | False alarm | Mean incidents |
|---|---|---|---|---|---|---|---|
| stable | none | n/a | — | 100% | 100% | 0% | 0.1 |
| benign_covariate | benign_covariate_shift | n/a | — | 100% | 100% | 0% | 1.4 |
| lab_unit_change | unit_change | 100% | 0.0 | 100% | 100% | 0% | 1.1 |
| feed_outage | feed_outage | 100% | 0.0 | 100% | 100% | 0% | 1.1 |
| prevalence_shift | prevalence_shift | 100% | 4.0 | 100% | 100% | 0% | 1.0 |
| concept_shift | concept_shift | 100% | 4.0 | 100% | 100% | 0% | 1.8 |
| subgroup_site | subgroup_degradation | 100% | 3.0 | 100% | 100% | 0% | 2.0 |

## Patient-level impact after the shift (harmful scenarios, totals across seeds; simulation oracle)

| Scenario | Events | Missed: no monitoring | Missed: ModelVigil | False alerts: no monitoring | False alerts: ModelVigil | Net benefit /1000: no monitoring | Net benefit /1000: ModelVigil |
|---|---|---|---|---|---|---|---|
| lab_unit_change | 6986 | 1624 | 2001 | 9831 | 6462 | 50.4 | 58.5 |
| feed_outage | 6986 | 2434 | 2187 | 5443 | 5955 | 55.4 | 57.5 |
| prevalence_shift | 10824 | 4178 | 3183 | 4467 | 7060 | 96.0 | 102.0 |
| concept_shift | 4876 | 1830 | 2169 | 8067 | 5975 | 17.9 | 21.1 |
| subgroup_site | 11278 | 4318 | 2773 | 5320 | 7977 | 97.7 | 113.0 |

Net benefit = TP/n - FP/n x pt/(1-pt) at the deployed 20% threshold (decision-curve analysis): it prices a false alert at 1/4 of a caught deterioration, the trade-off implied by the threshold.
