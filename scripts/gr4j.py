"""
GR4J rainfall-runoff model — pure NumPy implementation.

Based on:
    Perrin, C., Michel, C., & Andreassian, V. (2003).
    Improvement of a parsimonious model for streamflow simulation.
    Journal of Hydrology, 279(1-4), 275-289.

Designed as a physical-based baseline for LSTM rainfall-runoff models.
Companion module to lstm_gr4j_rainfall_runoff.ipynb.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution


# =============================================================================
# 1. PET — Hargreaves equation
# =============================================================================

def hargreaves_pet(tmin_c, tmax_c, lat_deg, dates):
    """
    Daily PET (mm/day) via Hargreaves (1985).

    PET = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)

    where Ra is extraterrestrial radiation (mm/day equivalent), computed from
    latitude and day-of-year. Requires only Tmin, Tmax, latitude, and date —
    no solar radiation observations.

    Parameters
    ----------
    tmin_c, tmax_c : np.ndarray
        Daily minimum and maximum temperature (degrees C).
    lat_deg : float
        Basin centroid latitude (degrees, positive = North).
    dates : pd.DatetimeIndex or array-like of datetime
        Dates corresponding to tmin/tmax (used for day-of-year).

    Returns
    -------
    pet : np.ndarray
        Daily PET (mm/day), non-negative.
    """
    tmin = np.asarray(tmin_c, dtype=float)
    tmax = np.asarray(tmax_c, dtype=float)
    tmean = 0.5 * (tmin + tmax)
    tdiff = np.maximum(tmax - tmin, 0.0)  # guard against negative spread

    # Day of year
    doy = pd.DatetimeIndex(dates).dayofyear.values

    # Extraterrestrial radiation Ra (FAO-56, Allen et al. 1998)
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)            # Earth-Sun distance
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)      # solar declination

    # Sunset hour angle (clip to avoid NaN at high latitudes / polar day-night)
    cos_ws = -np.tan(phi) * np.tan(delta)
    cos_ws = np.clip(cos_ws, -1.0, 1.0)
    ws = np.arccos(cos_ws)

    # Ra in MJ/m^2/day, then convert to equivalent mm/day (divide by 2.45)
    Gsc = 0.0820  # solar constant MJ/m^2/min
    Ra_MJ = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(phi) * np.sin(delta) +
        np.cos(phi) * np.cos(delta) * np.sin(ws)
    )
    Ra_mm = Ra_MJ / 2.45

    pet = 0.0023 * Ra_mm * (tmean + 17.8) * np.sqrt(tdiff)
    return np.maximum(pet, 0.0)


# =============================================================================
# 2. GR4J core simulation
# =============================================================================

def _build_unit_hydrographs(x4):
    """Discrete UH1 and UH2 ordinates for given X4 (days)."""
    nh1 = max(1, int(np.ceil(x4)))
    nh2 = max(1, int(np.ceil(2.0 * x4)))

    def SH1(t):
        if t <= 0:    return 0.0
        if t < x4:    return (t / x4) ** 2.5
        return 1.0

    def SH2(t):
        if t <= 0:        return 0.0
        if t <= x4:       return 0.5 * (t / x4) ** 2.5
        if t < 2 * x4:    return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
        return 1.0

    uh1 = np.array([SH1(i + 1) - SH1(i) for i in range(nh1)])
    uh2 = np.array([SH2(i + 1) - SH2(i) for i in range(nh2)])
    return uh1, uh2


def simulate_gr4j(P, E, params, S0_frac=0.6, R0_frac=0.7, return_states=False):
    """
    Run the GR4J daily rainfall-runoff model.

    Parameters
    ----------
    P : np.ndarray, shape (n,)
        Daily precipitation (mm/day).
    E : np.ndarray, shape (n,)
        Daily potential evapotranspiration (mm/day).
    params : array-like of length 4 — [X1, X2, X3, X4]
        X1: production store capacity (mm), typical 100-1200
        X2: groundwater exchange coefficient (mm/day), typical -5 to +3
        X3: routing store capacity (mm), typical 20-300
        X4: unit hydrograph time base (days), typical 1.1-2.9
    S0_frac, R0_frac : float
        Initial fill fractions for production and routing stores.
    return_states : bool
        If True, also return daily S and R series (useful for hybrid modeling).

    Returns
    -------
    Q_sim : np.ndarray (mm/day)
    states (optional) : dict with 'S' and 'R' arrays
    """
    X1, X2, X3, X4 = params
    P = np.asarray(P, dtype=float)
    E = np.asarray(E, dtype=float)
    n = len(P)

    # Initial states
    S = S0_frac * X1
    R = R0_frac * X3

    # Unit hydrographs
    UH1, UH2 = _build_unit_hydrographs(X4)
    Q9_state = np.zeros(len(UH1))
    Q1_state = np.zeros(len(UH2))

    Q_sim = np.zeros(n)
    if return_states:
        S_arr = np.zeros(n)
        R_arr = np.zeros(n)

    for t in range(n):
        Pt, Et = P[t], E[t]

        # 1) Net rainfall / net PET — only one is non-zero
        if Pt >= Et:
            Pn = Pt - Et
            # Fraction Ps of Pn that fills production store
            tnh = np.tanh(Pn / X1)
            Ps = X1 * (1 - (S / X1) ** 2) * tnh / (1 + (S / X1) * tnh)
            S = S + Ps
            Pr = Pn - Ps  # spillover
        else:
            En = Et - Pt
            tnh = np.tanh(En / X1)
            Es = S * (2 - S / X1) * tnh / (1 + (1 - S / X1) * tnh)
            S = S - Es
            Pr = 0.0

        # 2) Percolation from production store
        Perc = S * (1 - (1 + (4 / 9 * S / X1) ** 4) ** -0.25)
        S = S - Perc
        Pr = Pr + Perc  # total runoff entering UH branches

        # 3) Split 90% / 10%
        Pr9 = 0.9 * Pr
        Pr1 = 0.1 * Pr

        # 4) UH convolution (shift-and-add)
        Q9_state = np.roll(Q9_state, -1); Q9_state[-1] = 0.0
        Q9_state += Pr9 * UH1
        Q1_state = np.roll(Q1_state, -1); Q1_state[-1] = 0.0
        Q1_state += Pr1 * UH2
        Q9 = Q9_state[0]
        Q1 = Q1_state[0]

        # 5) Groundwater exchange
        F = X2 * (R / X3) ** 3.5

        # 6) Routing store
        R = max(0.0, R + Q9 + F)
        Qr = R * (1 - (1 + (R / X3) ** 4) ** -0.25)
        R = R - Qr

        # 7) Direct flow with same exchange term
        Qd = max(0.0, Q1 + F)

        Q_sim[t] = Qr + Qd
        if return_states:
            S_arr[t] = S
            R_arr[t] = R

    if return_states:
        return Q_sim, {'S': S_arr, 'R': R_arr}
    return Q_sim


# =============================================================================
# 2.5. Degree-day snow module (temperature-index)
# =============================================================================
# Pure GR4J treats every mm of precipitation as instant rainfall, which fails
# in snow-influenced basins: winter snowfall produces phantom runoff and the
# spring snowmelt peak is missed. We pre-process P with a simple two-threshold
# degree-day model before feeding GR4J.
#
# Phase separation and melt are decoupled via two independent thresholds:
#     T <  t_snow  ->  precipitation accumulates as snow (SWE += P)
#     T >= t_snow  ->  precipitation falls as rain
#     T >  t_melt  ->  snowpack melts at rate ddf * (T - t_melt), capped by SWE
#
# Effective rainfall fed to GR4J = rain (today's P if T >= t_snow) + meltwater.


def degree_day_snow(P, T, t_snow, t_melt, ddf, swe0=0.0, return_state=False):
    """
    Convert daily precipitation into effective rainfall using a temperature-
    index degree-day snow model.

    Parameters
    ----------
    P : np.ndarray, shape (n,)
        Daily precipitation (mm/day).
    T : np.ndarray, shape (n,)
        Daily mean air temperature (degrees C).
    t_snow : float
        Phase threshold (degrees C). Below this, precipitation falls as snow.
    t_melt : float
        Melt onset threshold (degrees C). Above this, snowpack melts.
    ddf : float
        Degree-day melt factor (mm / degC / day).
    swe0 : float
        Initial snow water equivalent (mm).
    return_state : bool
        If True, also return the daily SWE timeseries.

    Returns
    -------
    P_eff : np.ndarray (mm/day)
        Effective rainfall = liquid precipitation + snowmelt.
    states (optional) : dict with 'SWE' array.
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    n = len(P)

    P_eff = np.zeros(n)
    if return_state:
        SWE_arr = np.zeros(n)

    swe = float(swe0)
    for t in range(n):
        Pt, Tt = P[t], T[t]

        # 1) Phase separation
        if Tt < t_snow:
            snowfall, rain = Pt, 0.0
        else:
            snowfall, rain = 0.0, Pt
        swe += snowfall

        # 2) Degree-day melt (only if warm enough AND snowpack exists)
        if Tt > t_melt and swe > 0.0:
            melt = min(ddf * (Tt - t_melt), swe)
        else:
            melt = 0.0
        swe -= melt

        P_eff[t] = rain + melt
        if return_state:
            SWE_arr[t] = swe

    if return_state:
        return P_eff, {'SWE': SWE_arr}
    return P_eff


def simulate_gr4j_snow(P, T, E, params, S0_frac=0.6, R0_frac=0.7,
                       SWE0=0.0, return_states=False):
    """
    GR4J with a degree-day snow pre-processor (7 parameters total).

    The snow module converts (P, T) into effective rainfall before it enters
    the GR4J production store. For non-snow basins, set t_snow very low so
    that no precipitation is intercepted as snow — the model collapses to
    pure GR4J.

    Parameters
    ----------
    P : np.ndarray (mm/day)
        Daily precipitation.
    T : np.ndarray (degrees C)
        Daily mean air temperature (typically (Tmax + Tmin) / 2).
    E : np.ndarray (mm/day)
        Daily potential evapotranspiration.
    params : array-like of length 7 — [X1, X2, X3, X4, t_snow, t_melt, ddf]
        X1, X2, X3, X4 : standard GR4J parameters (see simulate_gr4j).
        t_snow : phase threshold (degC); T < t_snow => precip is snow.
        t_melt : melt onset threshold (degC); T > t_melt => melt.
        ddf    : degree-day melt factor (mm / degC / day).
    S0_frac, R0_frac : float
        Initial fill fractions for production and routing stores.
    SWE0 : float
        Initial snow water equivalent (mm).
    return_states : bool
        If True, also return daily S, R, SWE series.

    Returns
    -------
    Q_sim : np.ndarray (mm/day)
    states (optional) : dict with 'S', 'R', 'SWE' arrays.
    """
    X1, X2, X3, X4, t_snow, t_melt, ddf = params

    if return_states:
        P_eff, snow_states = degree_day_snow(
            P, T, t_snow, t_melt, ddf, swe0=SWE0, return_state=True)
        Q_sim, gr_states = simulate_gr4j(
            P_eff, E, [X1, X2, X3, X4],
            S0_frac=S0_frac, R0_frac=R0_frac, return_states=True)
        return Q_sim, {**gr_states, **snow_states}

    P_eff = degree_day_snow(P, T, t_snow, t_melt, ddf, swe0=SWE0)
    return simulate_gr4j(
        P_eff, E, [X1, X2, X3, X4], S0_frac=S0_frac, R0_frac=R0_frac)


# =============================================================================
# 3. Metrics (mirror the LSTM notebook for consistency)
# =============================================================================

def nse(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def rmse(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    return float(np.sqrt(np.mean((obs[mask] - sim[mask]) ** 2)))


def pbias(obs, sim):
    obs, sim = np.asarray(obs).flatten(), np.asarray(sim).flatten()
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[mask], sim[mask]
    return 100.0 * np.sum(sim - obs) / np.sum(obs)


# =============================================================================
# 4. Calibration
# =============================================================================

# Default bounds (Perrin et al. 2003 + safety margin)
DEFAULT_BOUNDS = [
    (10.0, 2000.0),   # X1 production capacity
    (-10.0, 10.0),    # X2 exchange coefficient
    (10.0,  500.0),   # X3 routing capacity
    (0.5,   10.0),    # X4 UH time base
]


def _objective_neg_nse(params, P, E, Q_obs, warmup):
    try:
        Q_sim = simulate_gr4j(P, E, params)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return 1e6
    if np.any(~np.isfinite(Q_sim)):
        return 1e6
    score = nse(Q_obs[warmup:], Q_sim[warmup:])
    if not np.isfinite(score):
        return 1e6
    return -score


def calibrate_gr4j(P, E, Q_obs, warmup=365, bounds=None, seed=42,
                   maxiter=200, workers=1, verbose=True):
    """
    Calibrate GR4J parameters by maximizing NSE on the warmup-trimmed series.

    Parameters
    ----------
    P, E, Q_obs : np.ndarray (mm/day)
    warmup : int
        Days to discard from objective evaluation (model spin-up).
    bounds : list of 4 (low, high) tuples or None
    seed : int
    maxiter : int
    workers : int
        scipy differential_evolution `workers`. Use 1 for reproducibility,
        -1 for parallel speed-up. (Parallel changes RNG behavior.)
    verbose : bool

    Returns
    -------
    result : dict with keys
        'params'        : np.ndarray [X1, X2, X3, X4]
        'nse_train'     : float
        'success'       : bool
        'n_iterations'  : int
        'raw'           : scipy OptimizeResult
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS

    P = np.asarray(P, dtype=float)
    E = np.asarray(E, dtype=float)
    Q_obs = np.asarray(Q_obs, dtype=float)

    res = differential_evolution(
        _objective_neg_nse,
        bounds=bounds,
        args=(P, E, Q_obs, warmup),
        seed=seed,
        maxiter=maxiter,
        polish=True,
        tol=1e-7,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate',
    )

    out = {
        'params': res.x,
        'nse_train': -res.fun,
        'success': res.success,
        'n_iterations': res.nit,
        'raw': res,
    }
    if verbose:
        x1, x2, x3, x4 = res.x
        print(f"  Calibration {'OK' if res.success else 'WARNING'} "
              f"(iters={res.nit}, NSE_train={-res.fun:.4f})")
        print(f"    X1 = {x1:8.2f}  (production capacity, mm)")
        print(f"    X2 = {x2:8.3f}  (exchange coefficient, mm/day)")
        print(f"    X3 = {x3:8.2f}  (routing capacity, mm)")
        print(f"    X4 = {x4:8.3f}  (UH time base, days)")
    return out


# =============================================================================
# 5. Calibration — GR4J + degree-day snow (7 parameters)
# =============================================================================
# GR4J core bounds (X1..X4) follow the Perrin et al. 2003 ranges with a safety
# margin, matching DEFAULT_BOUNDS. Earlier narrower bounds caused X1 and X3 to
# saturate at the upper edge for snow-dominated PNW basins (e.g. 13340600);
# widening lets the calibrator find a physically realistic deeper storage.
# Snow parameter bounds are kept moderate so the snow module remains
# identifiable rather than absorbing routing dynamics.

DEFAULT_BOUNDS_SNOW = [
    ( 10.0, 2000.0),  # X1     production capacity (mm)
    (-10.0,   10.0),  # X2     exchange coefficient (mm/day)
    ( 10.0,  500.0),  # X3     routing capacity (mm)
    (  0.5,   10.0),  # X4     UH time base (days)
    ( -3.0,    3.0),  # t_snow phase threshold (degC)
    (  0.0,    5.0),  # t_melt melt onset (degC)
    (  1.0,    8.0),  # ddf    degree-day factor (mm/degC/day)
]


def _objective_neg_nse_snow(params, P, T, E, Q_obs, warmup):
    try:
        Q_sim = simulate_gr4j_snow(P, T, E, params)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return 1e6
    if np.any(~np.isfinite(Q_sim)):
        return 1e6
    score = nse(Q_obs[warmup:], Q_sim[warmup:])
    if not np.isfinite(score):
        return 1e6
    return -score


def calibrate_gr4j_snow(P, T, E, Q_obs, warmup=365, bounds=None, seed=42,
                        maxiter=200, workers=1, verbose=True):
    """
    Calibrate GR4J + degree-day snow (7 parameters) by maximising NSE on the
    warmup-trimmed series.

    Parameters
    ----------
    P, E, Q_obs : np.ndarray (mm/day)
    T : np.ndarray (degrees C)
        Daily mean air temperature.
    warmup : int
        Days to discard from objective evaluation (model + SWE spin-up).
    bounds : list of 7 (low, high) tuples or None
        Defaults to DEFAULT_BOUNDS_SNOW.
    seed, maxiter, workers, verbose : see calibrate_gr4j.

    Returns
    -------
    result : dict with keys
        'params'        : np.ndarray [X1, X2, X3, X4, t_snow, t_melt, ddf]
        'nse_train'     : float
        'success'       : bool
        'n_iterations'  : int
        'raw'           : scipy OptimizeResult
    """
    if bounds is None:
        bounds = DEFAULT_BOUNDS_SNOW

    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=float)
    Q_obs = np.asarray(Q_obs, dtype=float)

    res = differential_evolution(
        _objective_neg_nse_snow,
        bounds=bounds,
        args=(P, T, E, Q_obs, warmup),
        seed=seed,
        maxiter=maxiter,
        polish=True,
        tol=1e-7,
        workers=workers,
        updating='deferred' if workers != 1 else 'immediate',
    )

    out = {
        'params': res.x,
        'nse_train': -res.fun,
        'success': res.success,
        'n_iterations': res.nit,
        'raw': res,
    }
    if verbose:
        x1, x2, x3, x4, ts, tm, ddf = res.x
        print(f"  Calibration {'OK' if res.success else 'WARNING'} "
              f"(iters={res.nit}, NSE_train={-res.fun:.4f})")
        print(f"    X1     = {x1:8.2f}  (production capacity, mm)")
        print(f"    X2     = {x2:8.3f}  (exchange coefficient, mm/day)")
        print(f"    X3     = {x3:8.2f}  (routing capacity, mm)")
        print(f"    X4     = {x4:8.3f}  (UH time base, days)")
        print(f"    t_snow = {ts:8.3f}  (phase threshold, degC)")
        print(f"    t_melt = {tm:8.3f}  (melt onset, degC)")
        print(f"    ddf    = {ddf:8.3f}  (degree-day factor, mm/degC/day)")
    return out
