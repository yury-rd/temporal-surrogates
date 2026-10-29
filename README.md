# Temporal surrogates

Code and saved results for *Temporal Structure in Resting-State fMRI Under Marginal and Cross-Spectral Surrogate Controls* by Aleksandr Volkov, Yury Maximov, Tamoghna Chattopadhyay, Paul M. Thompson, and Ildar Gabitov.

Manuscript for the IEEE International Symposium on Biomedical Imaging (ISBI).

The code compares sample entropy and finite MFDFA and Higuchi curve shapes with IAAFT and multichannel common-phase references. It also applies IAAFT to its own output to examine the behavior of the surrogate algorithm. The repository includes a runnable synthetic example, three human aggregate result files, and the complete synthetic observation-control rank table. Reproducing the participant-level ADNI analyses requires authorized data that are not included here.

## Run the example

Use Python 3.11 or later. Clone the repository, then install NumPy, SciPy, and threadpoolctl in a virtual environment:

```sh
git clone https://github.com/yury-rd/temporal-surrogates.git
cd temporal-surrogates
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --output results/demo.json
```

For the recorded package versions, install `requirements-repro.txt` instead of `requirements.txt`.
On Windows, activate the environment with `.venv\Scripts\activate`.

The example uses 197 samples from seven correlated stationary Gaussian channels at a sampling interval of 1.5 seconds. It evaluates 20 references per family, with IAAFT budgets of 25 and 250 iterations, and repeats the comparison on a fixed IAAFT output. The JSON contains observed descriptors, per-channel ranks, and measured constraint errors. Random seeds are fixed in the implementation.

Read the saved human aggregate results without recomputing measurements:

```sh
python run.py --mode summary --output results/aggregates.json
```

Omit `--output` to print JSON to standard output. The example does not rerun the full human analysis or all 1,280 synthetic control conditions.

## Use local network time series

Provide an NPZ file with a `timeseries` array of shape `samples × 7`. Values must be finite real numbers, with at least 64 samples. For an existing NumPy array `networks`, save it with `np.savez("local_networks.npz", timeseries=networks)`.

```sh
python run.py --input local_networks.npz --output results/local_surrogates.json
```

Retain the native sampling interval and sample support when interpreting the curves. Scales and lags are measured in samples; this command does not resample the input or infer its TR. The human analysis used equally weighted Schaefer parcels to form seven network means after removal of leading non-steady-state frames, before nuisance regression.

Short records, constant channels, and sparse template matches can leave a descriptor unsupported even when the input passes validation. Unsupported ranks and undefined numeric outputs are represented by JSON `null`, with descriptor status and reason fields where available.

## Interpretation and saved results

IAAFT preserves the empirical marginal exactly and approximates Fourier amplitudes at a fixed iteration budget. Common-phase references preserve the complete discrete cross-spectrum and full-record functional connectivity, while allowing the marginals to change. A rank of 1 among 21 objects means the observation has a uniquely larger leave-one-out score than all 20 references. These ranks are descriptive comparisons under the selected reference constraints.

[`data/README.md`](data/README.md) describes the saved results and their scope. The implementation is in `surrogates.py`; `run.py` handles the example, local inputs, and aggregate summaries. Tests are in `tests/`.

Run the tests with `python -m unittest discover -s tests -v`. The runner finds its data relative to its own location, so it can also be called by its full path from another directory. Input and output paths are relative to the current working directory.

## Citation and license

Please cite the manuscript using [`CITATION.cff`](CITATION.cff) and link to [this repository](https://github.com/yury-rd/temporal-surrogates) when using the code or saved results. The software and included synthetic and aggregate results are distributed under the [MIT license](LICENSE). ADNI data remain subject to the applicable [ADNI access agreement](https://adni.loni.usc.edu/).
