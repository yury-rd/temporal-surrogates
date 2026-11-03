# Temporal surrogates

Code for *Temporal Structure in Resting-State fMRI Under Marginal and Cross-Spectral Surrogate Controls* by Aleksandr Volkov, Yury Maximov, Tamoghna Chattopadhyay, Paul M. Thompson, and Ildar Gabitov.

Manuscript for the IEEE International Symposium on Biomedical Imaging (ISBI).

The public code compares sample entropy and finite MFDFA and Higuchi curve
shapes with IAAFT and multichannel common-phase references. It also applies
IAAFT to its own output to examine the behavior of the surrogate algorithm.
Saved human aggregates and the saved synthetic observation-control rank table
are stored in the private
[yury-rd/temporal-surrogates-data](https://github.com/yury-rd/temporal-surrogates-data)
repository. The public repository references that private checkout through its
optional `data/` Git submodule; an ordinary clone does not download it.

## Run without private data

Use Python 3.11 or later:

```sh
git clone https://github.com/yury-rd/temporal-surrogates.git
cd temporal-surrogates
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-repro.txt
python run.py --output results/demo.json
python -m unittest discover -s tests -v
```

The pinned environment contains NumPy, SciPy, and threadpoolctl. For supported
version ranges, use `requirements.txt`. On Windows, activate the environment
with `.venv\Scripts\activate`.

The example generates 197 samples from seven correlated stationary Gaussian
channels at a sampling interval of 1.5 seconds. It evaluates 20 references per
family, with IAAFT budgets of 25 and 250 iterations, and repeats the comparison
on a fixed IAAFT output. The JSON contains observed descriptors, per-channel
ranks, and measured constraint errors. Seeds are fixed in the implementation.
The example and public tests generate their inputs at runtime and require no
private data or GitHub credentials. Omit `--output` to print JSON to stdout.

## Read saved results with authorized access

Access to the companion repository must be granted separately. With GitHub
credentials that have that access:

```sh
git submodule update --init data
python run.py --mode summary --output results/aggregates.json
```

The submodule pins a private data commit. Updating the public checkout alone
does not change or grant access to the private data. If access is unavailable,
the synthetic example and public tests remain usable.

An existing authorized data checkout can be selected explicitly:

```sh
python run.py --mode summary --data-dir /path/to/temporal-surrogates-data --output results/aggregates.json
```

The selected directory must contain these files directly, as on the companion's
`data` branch:

| File | Contents |
| --- | --- |
| `iaaft_aggregate.json` | Aggregate comparisons with univariate IAAFT references and spectral-constraint diagnostics. |
| `iaaft_calibration_aggregate.json` | IAAFT applied to its own output, compared with observations on matched support. |
| `common_phase_aggregate.json` | Aggregate comparisons with references preserving the seven-network discrete cross-spectrum. |
| `observation_support_ranks.csv` | Saved synthetic observation-control ranks; retained privately and not read by summary mode. |

Summary mode reads the three JSON files; it does not recompute the human
analyses or synthetic control panel. Missing files produce an error explaining
how to initialize the private submodule or use `--data-dir`. The companion's
README describes the saved cohort and result schemas.

## Use local network time series

Provide an NPZ file with a `timeseries` array of shape `samples × 7`. Values
must be finite real numbers, with at least 64 samples. For a NumPy array
`networks`, save it with `np.savez("local-data/networks.npz", timeseries=networks)`
after creating `local-data/`.

```sh
python run.py --input local-data/networks.npz --output results/local_surrogates.json
```

This mode requires only the supplied NPZ. Retain the native sampling interval
and sample support when interpreting curves. Scales and lags are measured in
samples; the command does not resample input or infer its TR. The human analysis
used equally weighted Schaefer parcels to form seven network means after removal
of leading non-steady-state frames, before nuisance regression.

Short records, constant channels, and sparse template matches can leave a
descriptor unsupported even when input passes validation. Unsupported ranks
and undefined numeric outputs are represented by JSON `null`, with descriptor
status and reason fields where available.

The runner locates the default private checkout relative to its own location.
`--input`, `--output`, and `--data-dir` paths are relative to the current working
directory. Keep local inputs under ignored `local-data/` and generated outputs
under `results/`; do not commit either to the public code repository.

## Interpretation and implementation

IAAFT preserves the empirical marginal exactly and approximates Fourier
amplitudes at a fixed iteration budget. Common-phase references preserve the
complete discrete cross-spectrum and full-record functional connectivity while
allowing marginals to change. A rank of 1 among 21 objects means the observation
has a uniquely larger leave-one-out score than all 20 references. Ranks are
descriptive comparisons under the selected reference constraints.

`surrogates.py` contains the measurement, reference algorithms, and synthetic
generators. `run.py` handles the example, local inputs, and private aggregate
summaries. Tests verify Fourier constraints, descriptor calculations, input
contracts, command-line execution, and missing-data behavior using runtime
fixtures. The inactive [CI template](.github/check-template.yml) runs these
public tests without initializing the private submodule or fetching saved
results. To enable it, copy it to `.github/workflows/check.yml` and push with
credentials permitted to update workflows.

Reproducing participant-level ADNI analyses requires authorized data and the
corresponding analysis workflow beyond this package. Raw scans and
participant-level records are not distributed with the public code.

## Citation and license

Please cite the manuscript using [CITATION.cff](CITATION.cff) and link to
[this repository](https://github.com/yury-rd/temporal-surrogates) when using the
code. The software and synthetic generators are distributed under the
[MIT license](LICENSE). The private companion retains the terms for saved
results. ADNI data remain subject to the applicable
[ADNI access agreement](https://adni.loni.usc.edu/); private repository access
does not replace that agreement.
