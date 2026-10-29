# Saved results

The JSON files contain aggregate human comparisons from the paper's 270-participant, 732-visit cohort. Seven networks per visit give 5,124 network-visits. These files contain aggregate results without participant-level time series.

| File | Contents |
| --- | --- |
| `iaaft_aggregate.json` | Observed descriptors compared with univariate IAAFT references, including rank fractions and spectral-constraint diagnostics. |
| `iaaft_calibration_aggregate.json` | IAAFT applied to its own output, compared with the original observations on matched support. SampEn has 5,123 supported pairs; the curve shapes have 5,124. |
| `common_phase_aggregate.json` | Comparisons with common-phase references that preserve the seven-network discrete cross-spectrum, including rank fractions and constraint diagnostics. |
| `observation_support_ranks.csv` | The complete synthetic observation-control rank table for 1,280 conditions. |

To read all three JSON aggregates:

```sh
python run.py --mode summary --output results/aggregates.json
```

Run the command from the repository root. It reads the saved files; it does not recompute the human analyses.

## Synthetic observation controls

The CSV covers 32 independent Gaussian process realizations on ten sample-count/TR grids under four observation branches: identity, pointwise cube, a twofold mid-record gain step, and intercept-and-trend removal. Each condition retains seven channels, three descriptor families, and three reference families. Blank ranks denote unsupported measurements.

Its 80,640 rows use the columns `case_id`, `seed`, `N`, `TR_seconds`, `branch`, `family`, `metric`, `network_index`, `rank`, and `status`. Network indices run from 0 to 6. The reference families are `iaaft25`, `iaaft250`, and `common_phase`; the metrics are `SampEn`, `MFDFA_shape`, and `Higuchi_shape`.

The grids cross sample counts 97, 197, and 480 with TRs of 0.607, 1.5, and 3 seconds, and add 193 samples at TR 1.5 seconds. Shared timestamps allow different grids to observe the same latent path. `latent_paths`, `cells`, `observation`, `iaaft_pair`, and `phase_draw` in `surrogates.py` provide the corresponding generators and observation operations. The default `run.py` example evaluates one synthetic grid and is a compact calibration demonstration.

## Human data access

Raw ADNI scans and participant-level derived records are not included. Obtain authorized data through [ADNI](https://adni.loni.usc.edu/) and follow its access agreement. The public aggregates support inspection of the reported comparisons; reproducing the human cohort selection, repeatability analysis, and clinical-change model requires the authorized inputs and corresponding analysis workflow.

The included synthetic and aggregate result files are distributed under the repository's [MIT license](../LICENSE).
