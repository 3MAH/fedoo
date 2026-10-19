"""Shared SO(3) helpers: skew matrices and the left Jacobian of the rotations."""

import numpy as np

# Below this angle, the coefficients use their Taylor series in angle**2: the
# closed forms of alpha, beta and gamma lose up to 8 digits to cancellation
# near 0.05 rad and are accurate to ~2e-14 from 1 rad on. Ten terms keep the
# series at round-off up to 1 rad (radius of convergence 2 pi).
_SERIES = 1.0
_SERIES_COEFFICIENTS = {
    "a": (
        1 / 12,
        1 / 720,
        1 / 30240,
        1 / 1209600,
        1 / 47900160,
        691 / 1307674368000,
        1 / 74724249600,
        3617 / 10670622842880000,
        43867 / 5109094217170944000,
        174611 / 802857662698291200000,
    ),
    "alpha": (
        1 / 360,
        1 / 7560,
        1 / 201600,
        1 / 5987520,
        691 / 130767436800,
        1 / 6227020800,
        3617 / 762187345920000,
        43867 / 319318388573184000,
        174611 / 44603203483238400000,
        77683 / 705055001969590272000,
    ),
    "b": (
        1 / 2,
        -1 / 24,
        1 / 720,
        -1 / 40320,
        1 / 3628800,
        -1 / 479001600,
        1 / 87178291200,
        -1 / 20922789888000,
        1 / 6402373705728000,
        -1 / 2432902008176640000,
    ),
    "beta": (
        -1 / 12,
        1 / 180,
        -1 / 6720,
        1 / 453600,
        -1 / 47900160,
        1 / 7264857600,
        -1 / 1494484992000,
        1 / 400148356608000,
        -1 / 135161222676480000,
        1 / 56200036388880384000,
    ),
    "c": (
        1 / 6,
        -1 / 120,
        1 / 5040,
        -1 / 362880,
        1 / 39916800,
        -1 / 6227020800,
        1 / 1307674368000,
        -1 / 355687428096000,
        1 / 121645100408832000,
        -1 / 51090942171709440000,
    ),
    "gamma": (
        -1 / 60,
        1 / 1260,
        -1 / 60480,
        1 / 4989600,
        -1 / 622702080,
        1 / 108972864000,
        -1 / 25406244864000,
        1 / 7602818775552000,
        -1 / 2838385676206080000,
        1 / 1292600836944248832000,
    ),
}


def skew(v):
    """Skew matrices for vectors with arbitrary leading batch dimensions."""
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    return np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(
        v.shape[:-1] + (3, 3)
    )


def _series(name, t2):
    value = np.zeros_like(t2)
    for coefficient in reversed(_SERIES_COEFFICIENTS[name]):
        value = value * t2 + coefficient
    return value


def jacobian_coefficients(angle):
    """Coefficients of J^-1 (a, alpha) and J (b, beta, c, gamma) of SO(3).

    J(p) = I + b [p] + c [p]^2 and J(p)^-1 = I - [p]/2 + a [p]^2 are the left
    Jacobian and its inverse; alpha, beta, gamma are a', b', c' over the angle.
    """
    t = np.atleast_1d(np.asarray(angle, dtype=float))
    t2 = t * t
    a, alpha, b, beta, c, gamma = (
        _series(name, t2) for name in ("a", "alpha", "b", "beta", "c", "gamma")
    )
    large = t >= _SERIES
    if np.any(large):
        x = t[large]
        cot = 1 / np.tan(x / 2)
        sin, cos = np.sin(x), np.cos(x)
        a[large] = (1 - 0.5 * x * cot) / x**2
        alpha[large] = (
            -2 / x**3 + cot / (2 * x**2) + 1 / (4 * x * np.sin(x / 2) ** 2)
        ) / x
        b[large] = (1 - cos) / x**2
        beta[large] = (x * sin - 2 * (1 - cos)) / x**4
        c[large] = (x - sin) / x**3
        gamma[large] = ((1 - cos) * x - 3 * (x - sin)) / x**5
    shape = np.shape(angle)
    return tuple(value.reshape(shape) for value in (a, alpha, b, beta, c, gamma))


def left_jacobian(rotvec):
    """Left Jacobian J(p) = I + b [p] + c [p]^2, batched over leading axes."""
    rotvec = np.asarray(rotvec, dtype=float)
    _, _, b, _, c, _ = jacobian_coefficients(np.linalg.norm(rotvec, axis=-1))
    s = skew(rotvec)
    return np.eye(3) + b[..., None, None] * s + c[..., None, None] * (s @ s)
