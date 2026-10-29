"""Finite temporal descriptors, surrogate generators, and observation controls."""

import numpy as np
from scipy.spatial import cKDTree


Q = np.array([-2.0, 0.0, 2.0, 4.0])
SCALES = np.array([8, 12, 16, 24, 32, 48])
ITER = 25


def higuchi(x: np.ndarray) -> dict:
    """Compute finite Higuchi lengths and their fitted slope."""
    n = len(x)
    if n < 64 or not np.isfinite(x).all() or np.std(x) < 1e-12:
        return {"status": "unsupported", "reason": "short_nonfinite_or_constant"}
    kmax = min(16, n // 8)
    curve = []
    for k in range(1, kmax + 1):
        lengths = []
        for m in range(k):
            z = x[m::k]
            jumps = len(z) - 1
            if jumps > 0:
                # The second k converts the polyline length to a unit time grid.
                lengths.append(
                    float(np.abs(np.diff(z)).sum() * (n - 1) / (jumps * k * k))
                )
        curve.append(float(np.mean(lengths)))
    v = np.asarray(curve)
    if not np.isfinite(v).all() or np.any(v <= 0):
        return {"status": "unsupported", "reason": "nonpositive_curve"}
    a = np.log(1.0 / np.arange(1, kmax + 1))
    b = np.log(v)
    slope, intercept = np.polyfit(a, b, 1)
    fit = slope * a + intercept
    denom = np.sum((b - b.mean()) ** 2)
    r2 = float(1 - np.sum((b - fit) ** 2) / denom) if denom > 0 else 1.0
    return {
        "status": "defined",
        "dimension": float(slope),
        "r2": r2,
        "k": list(range(1, kmax + 1)),
        "L": v.tolist(),
    }


def mfdfa(x: np.ndarray) -> dict:
    """Compute MFDFA curves using forward segments and linear detrending."""
    n = len(x)
    if n < 64 or not np.isfinite(x).all() or np.std(x) < 1e-12:
        return {"status": "unsupported", "reason": "short_nonfinite_or_constant"}
    y = np.cumsum(x - x.mean())
    fq = np.full((len(Q), len(SCALES)), np.nan)
    segments = []
    direct_f2 = []
    for si, s in enumerate(SCALES):
        count = n // int(s)
        segments.append(int(count))
        if count < 4:
            direct_f2.append(None)
            continue
        z = y[:count * s].reshape(count, s)
        t = np.arange(s, dtype=float) - (s - 1) / 2
        centered = z - z.mean(axis=1, keepdims=True)
        slope = centered @ t / float(t @ t)
        resid = centered - slope[:, None] * t
        v = np.mean(resid * resid, axis=1)
        v = v[v > 1e-24]
        if len(v) < 4:
            direct_f2.append(None)
            continue
        for qi, q in enumerate(Q):
            fq[qi, si] = (
                (np.mean(v ** (q / 2))) ** (1 / q)
                if q
                else np.exp(np.mean(np.log(v)) / 2)
            )
        direct_f2.append(float(np.sqrt(np.mean(v))))
    h = []
    for curve in fq:
        valid = np.isfinite(curve) & (curve > 0)
        h.append(
            float(np.polyfit(np.log(SCALES[valid]), np.log(curve[valid]), 1)[0])
            if valid.sum() >= 3
            else None
        )
    if any(v is None for v in h):
        return {
            "status": "unsupported",
            "reason": "insufficient_positive_scales",
            "scales": SCALES.tolist(),
            "segments": segments,
        }
    return {
        "status": "defined",
        "q": Q.tolist(),
        "scales": SCALES.tolist(),
        "segments": segments,
        "Fq": [
            [None if not np.isfinite(v) else float(v) for v in row] for row in fq
        ],
        "h": h,
        "contrast_h2_minus_hminus2": h[2] - h[0],
        "direct_F2": direct_f2,
    }


def sampen(x: np.ndarray) -> dict:
    """Compute sample entropy with explicit pairwise template comparisons."""
    n = len(x)
    if n < 64 or not np.isfinite(x).all():
        return {"status": "unsupported", "reason": "short_or_nonfinite"}
    tolerance = 0.2 * float(np.std(x))
    if tolerance <= 1e-12:
        return {"status": "unsupported", "reason": "constant"}
    # The two- and three-point templates use the same set of start positions.
    z = np.lib.stride_tricks.sliding_window_view(x, 3)
    d0 = np.abs(z[:, None, 0] - z[None, :, 0])
    d1 = np.abs(z[:, None, 1] - z[None, :, 1])
    d2 = np.abs(z[:, None, 2] - z[None, :, 2])
    upper = np.triu(np.ones((len(z), len(z)), dtype=bool), k=1)
    matches2 = upper & (d0 <= tolerance) & (d1 <= tolerance)
    b = int(np.sum(matches2))
    a = int(np.sum(matches2 & (d2 <= tolerance)))
    if not b or not a:
        return {
            "status": "unsupported",
            "reason": "zero_template_matches",
            "A": a,
            "B": b,
            "tolerance": tolerance,
        }
    return {
        "status": "defined",
        "value": float(-np.log(a / b)),
        "A": a,
        "B": b,
        "tolerance": tolerance,
    }


def iaaft(x, rng):
    """Generate an IAAFT reference with the fixed 25-iteration budget."""
    sorted_x = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(ITER):
        projected = np.fft.irfft(
            amp * np.exp(1j * np.angle(np.fft.rfft(y))), n=len(x)
        )
        y = np.empty_like(x)
        y[np.argsort(projected, kind="stable")] = sorted_x
    error = float(np.linalg.norm(np.abs(np.fft.rfft(y)) - amp) / np.linalg.norm(amp))
    if not np.array_equal(np.sort(y), sorted_x):
        raise RuntimeError("IAAFT marginal failure")
    return y, error


def sampen_fast(x):
    """Count the same sample-entropy matches using a spatial tree."""
    n = len(x)
    tol = 0.2 * float(np.std(x))
    if n < 64 or tol <= 1e-12 or not np.isfinite(x).all():
        return {"status": "unsupported", "reason": "short_nonfinite_or_constant"}
    z = np.lib.stride_tricks.sliding_window_view(x, 3)

    def count(d):
        tree = cKDTree(z[:, :d])
        return int((tree.count_neighbors(tree, tol, p=np.inf) - len(z)) // 2)

    b = count(2)
    a = count(3)
    if not b or not a:
        return {
            "status": "unsupported",
            "reason": "zero_template_matches",
            "A": a,
            "B": b,
        }
    return {
        "status": "defined",
        "A": a,
        "B": b,
        "value": float(-np.log(a / b)),
        "tolerance": tol,
    }


def mfshape(obj):
    """Return the 12-coordinate MFDFA shape, or None if unsupported."""
    if obj["status"] != "defined":
        return None
    fq = np.array([[row[i] for i in (0, 1, 2, 3)] for row in obj["Fq"]], float)
    if fq.shape != (4, 4) or np.any(fq <= 0) or not np.isfinite(fq).all():
        return None
    return np.log(fq[:, 1:] / fq[:, [0]]).ravel()


def hshape(obj):
    """Return the 11-coordinate Higuchi shape, or None if unsupported."""
    if obj["status"] != "defined":
        return None
    z = np.asarray(obj["L"][:12], float)
    if len(z) != 12 or np.any(z <= 0) or not np.isfinite(z).all():
        return None
    return np.log(z[1:] / z[0])


def rank_stat(values):
    """Score each object against the others and rank the first object."""
    if not np.isfinite(values).all():
        return None, None
    scores = []
    for i in range(len(values)):
        others = np.concatenate((values[:i], values[i + 1:]), axis=0)
        med = np.median(others, axis=0)
        sd = np.maximum(others.std(axis=0), 1e-6)
        scores.append(float(np.sqrt(np.mean(((values[i] - med) / sd) ** 2))))
    rank = 1 + sum(v >= scores[0] for v in scores[1:])
    return rank, scores[0]


def phase_draw(net, rng):
    """Randomize common phases and record cross-spectrum, FC, and marginal errors."""
    n = len(net)
    f = np.fft.rfft(net, axis=0)
    angles = rng.uniform(-np.pi, np.pi, len(f))
    phase = np.exp(1j * angles)
    phase[0] = 1
    if n % 2 == 0:
        phase[-1] = rng.choice(np.array([-1.0, 1.0]))
    y = np.fft.irfft(f * phase[:, None], n=n, axis=0)
    yf = np.fft.rfft(y, axis=0)
    cross = np.einsum("fi,fj->fij", f, np.conj(f))
    newcross = np.einsum("fi,fj->fij", yf, np.conj(yf))
    err = float(np.linalg.norm(newcross - cross) / np.linalg.norm(cross))
    fc = np.corrcoef(net, rowvar=False)
    fc2 = np.corrcoef(y, rowvar=False)
    fcerr = float(np.max(np.abs(fc - fc2)))
    marginal = np.array([
        np.sqrt(np.mean((np.sort(y[:, j]) - np.sort(net[:, j])) ** 2))
        / np.std(net[:, j])
        for j in range(7)
    ])
    return y, err, fcerr, marginal


def iaaft_pair(x, rng):
    """Save the 25-step iterate, then continue the same draw to 250 iterations."""
    sx = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    states = {}
    for iteration in range(1, 251):
        projected = np.fft.irfft(
            amp * np.exp(1j * np.angle(np.fft.rfft(y))), n=len(x)
        )
        y = np.empty_like(x)
        y[np.argsort(projected, kind="stable")] = sx
        if iteration in (25, 250):
            assert np.array_equal(np.sort(y), sx)
            err = float(
                np.linalg.norm(np.abs(np.fft.rfft(y)) - amp) / np.linalg.norm(amp)
            )
            states[iteration] = (y.copy(), err)
    return states


def observation(x, branch):
    """Apply an observation change to a latent multichannel series."""
    if branch == "identity":
        return x.copy()
    if branch == "cubic":
        return x ** 3
    if branch == "gain":
        y = x.copy()
        y[len(x) // 2:] *= 2
        return y
    if branch == "detrend":
        t = np.arange(len(x), dtype=float)
        t -= t.mean()
        z = x - x.mean(axis=0)
        return z - t[:, None] * (t @ z / (t @ t))[None, :]
    raise ValueError(branch)


def rng_for(*seed):
    """Construct the study's PCG64 generator from an integer seed sequence."""
    return np.random.Generator(np.random.PCG64(list(seed)))


def cells():
    """Return sample counts and sampling intervals in milliseconds."""
    return [
        (n, tr) for n in (97, 197, 480) for tr in (607, 1500, 3000)
    ] + [(193, 1500)]


def latent_paths(seed):
    """Generate stationary Gaussian transitions on the union of sampling grids."""
    ts = np.unique(np.concatenate([
        np.arange(n, dtype=np.int64) * tr for n, tr in cells()
    ]))
    rng = rng_for(seed)
    innov = rng.standard_normal((len(ts), 8), dtype=np.float64)
    a = -np.log(0.7) / 3.0
    z = np.empty_like(innov)
    z[0] = innov[0]
    for i in range(1, len(ts)):
        dt = float(ts[i] - ts[i - 1]) / 1000.0
        phi = np.exp(-a * dt)
        z[i] = phi * z[i - 1] + np.sqrt(1 - np.exp(-2 * a * dt)) * innov[i]
    y = (z[:, :7] + z[:, [7]]) / np.sqrt(2.0)
    return ts, z, {
        (n, tr): y[np.searchsorted(ts, np.arange(n, dtype=np.int64) * tr)]
        for n, tr in cells()
    }


def measurement(x):
    """Evaluate the three descriptors for a single channel."""
    return {"SampEn": sampen_fast(x), "MFDFA": mfdfa(x), "Higuchi": higuchi(x)}


def vectors(z):
    """Extract descriptor vectors, retaining unsupported measurements as None."""
    e = z["SampEn"]
    return {
        "SampEn": np.array([e["value"]]) if e["status"] == "defined" else None,
        "MFDFA_shape": mfshape(z["MFDFA"]),
        "Higuchi_shape": hshape(z["Higuchi"]),
    }
