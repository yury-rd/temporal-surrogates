"""Run the seven-channel surrogate demonstration or read saved aggregates."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import surrogates as m


ROOT = Path(__file__).resolve().parent


def plain(value):
    """Convert NumPy values to JSON types; undefined numeric values become null."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def validate_timeseries(timeseries):
    """Validate N-by-7 real data with N >= 64 and normalize it to float64.

    Floating-point normalization prevents unsigned integer differences from
    wrapping and avoids running the descriptors in low-precision input dtypes.
    The default synthetic fixture is already float64 and is unchanged.
    """
    x = np.asarray(timeseries)
    if x.ndim != 2 or x.shape[1] != 7:
        raise ValueError("Input timeseries must have shape N-by-7")
    if x.shape[0] < 64:
        raise ValueError("Input timeseries must contain at least 64 samples")
    if x.dtype.kind not in "iuf":
        raise ValueError("Input timeseries must contain real numeric values")
    if not np.isfinite(x).all():
        raise ValueError("Input timeseries must contain only finite values")
    with np.errstate(over="ignore"):
        x = x.astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Input timeseries must be representable as finite float64 values")
    return x


def load_timeseries(path):
    """Read a local NPZ containing a real numeric ``timeseries`` array."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "timeseries" not in archive:
                raise ValueError("Input NPZ must contain a 'timeseries' array")
            x = archive["timeseries"]
    except (TypeError, AttributeError) as error:
        raise ValueError("Input must be an NPZ containing a 'timeseries' array") from error
    return validate_timeseries(x)


def surrogate_demo(timeseries=None):
    """Evaluate the fixed demonstration or a supplied N-by-7 observation."""
    if timeseries is None:
        _, _, paths = m.latent_paths(2026102201)
        x = paths[(197, 1500)]
    else:
        x = validate_timeseries(timeseries)
    observed = [m.measurement(x[:, j]) for j in range(7)]
    pseudo = np.column_stack([
        m.iaaft(x[:, j], m.rng_for(2026102201, j, 100))[0] for j in range(7)
    ])
    out = {}
    for label, field in [("observed", x), ("iaaft_as_observation", pseudo)]:
        bank = {
            k: [[m.measurement(field[:, j]) for j in range(7)]]
            for k in ["iaaft25", "iaaft250", "common_phase"]
        }
        errors = {"iaaft25": [], "iaaft250": [], "common_phase": []}
        for draw in range(20):
            pair = [
                m.iaaft_pair(
                    field[:, j], m.rng_for(2026102201, len(x), 1500, draw, j, 310)
                )
                for j in range(7)
            ]
            for budget, name in [(25, "iaaft25"), (250, "iaaft250")]:
                bank[name].append([m.measurement(v[budget][0]) for v in pair])
                errors[name].extend(v[budget][1] for v in pair)
            shifted, cross, fc, _ = m.phase_draw(
                field, m.rng_for(2026102201, len(x), 1500, draw, 410)
            )
            bank["common_phase"].append([
                m.measurement(shifted[:, j]) for j in range(7)
            ])
            errors["common_phase"].append(max(cross, fc))
        ranks = {}
        for family, draws in bank.items():
            ranks[family] = {}
            for metric in ["SampEn", "MFDFA_shape", "Higuchi_shape"]:
                result = []
                for channel in range(7):
                    values = [m.vectors(row[channel])[metric] for row in draws]
                    result.append(
                        m.rank_stat(np.stack(values))[0]
                        if all(v is not None for v in values)
                        else None
                    )
                ranks[family][metric] = result
        out[label] = {
            "ranks": ranks,
            "maximum_constraint_error": {k: max(v) for k, v in errors.items()},
        }
    return {
        "N": len(x),
        "surrogates": 20,
        "observed_descriptors": observed,
        "calibration": "IAAFT applied to its own output; finite generator diagnostic",
        "readouts": out,
    }


def summary(data_dir=None):
    """Read the saved JSON aggregates without recomputing measurements."""
    folder = ROOT / "data" if data_dir is None else Path(data_dir)
    return {
        name: json.loads((folder / name).read_text())
        for name in (
            "common_phase_aggregate.json",
            "iaaft_aggregate.json",
            "iaaft_calibration_aggregate.json",
        )
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["demo", "summary"], default="demo")
    parser.add_argument("--input", type=Path, help="Local NPZ with an N-by-7 timeseries array")
    parser.add_argument("--output", type=Path, help="JSON output path; defaults to stdout")
    args = parser.parse_args(argv)
    if args.mode == "summary" and args.input:
        parser.error("--input is only available in demo mode")
    try:
        if args.mode == "summary":
            result = summary()
        else:
            x = load_timeseries(args.input) if args.input else None
            # Undefined constraints for constant channels are serialized as null.
            with threadpool_limits(limits=1), np.errstate(divide="ignore", invalid="ignore"):
                result = surrogate_demo(x)
        payload = json.dumps(plain(result), indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
            print(json.dumps({
                "output": str(args.output),
                "mode": args.mode,
            }))
        else:
            print(payload, end="")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
