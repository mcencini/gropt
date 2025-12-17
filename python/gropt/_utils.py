"""Gradient waveform utilities for eddy currents, diffusion, flow compensation, and plotting."""

__all__ = [
    'asymmbipolar_diffusion',
    'bipolar_diffusion',
    'conventional_flowcomp',
    'conventional_flowencode',
    'get2_eddy_mode0',
    'get2_eddy_mode1',
    'get_bval',
    'get_eddy_curves',
    'get_min_TE',
    'get_min_TE_diff',
    'get_min_TE_free',
    'get_moment_plots',
    'get_moments',
    'get_stim',
    'monopolar_diffusion',
    'plot_moments',
    'plot_waveform',
]

from math import ceil
from typing import Any, MutableMapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

from ._gropt import gropt


def get2_eddy_mode0(
    G: NDArray[np.floating], lam: float, dt: float
) -> NDArray[np.floating]:
    """
    Compute zeroth-order eddy current response for a gradient waveform.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform (1D). Only the length is used here.
    lam : float
        Eddy current time constant in seconds.
    dt : float
        Sample spacing of `G` in seconds.

    Returns
    -------
    NDArray[np.floating]
        Zeroth-order eddy response array with the same shape as `G`.
    """
    E0 = np.zeros_like(G)
    for i in range(G.size):
        ii = float(G.size - i - 1)
        if i == 0:
            val = -np.exp(-ii * dt / lam)
        else:
            val = np.exp(-(ii + 1.0) * dt / lam) - np.exp(-ii * dt / lam)
        E0[i] = -val

    return E0


def get2_eddy_mode1(
    G: NDArray[np.floating], lam: float, dt: float
) -> NDArray[np.floating]:
    """
    Compute first-order eddy current response for a gradient waveform.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform (1D). Only the length is used here.
    lam : float
        Eddy current time constant in seconds.
    dt : float
        Sample spacing of `G` in seconds.

    Returns
    -------
    NDArray[np.floating]
        First-order eddy response array with the same shape as `G`.
    """
    E1 = np.zeros_like(G)

    val = 0.0
    for i in range(G.size):
        ii = float(G.size - i - 1)
        val += -np.exp(-ii * dt / lam)
    E1[0] = val * 1e3 * dt

    for i in range(1, G.size):
        ii = float(G.size - i)
        val = -np.exp(-ii * dt / lam)
        E1[i] = val * 1e3 * dt

    return E1


def get_eddy_curves(
    G: NDArray[np.floating],
    dt: float,
    max_lam: float,
    n_lam: int,
) -> tuple[NDArray[np.floating], list[float], list[float]]:
    """
    Sweep eddy current responses over a range of time constants.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform (1D).
    dt : float
        Sample spacing of `G` in seconds.
    max_lam : float
        Maximum lambda in milliseconds to sweep (converted internally to seconds).
    n_lam : int
        Number of lambda samples.

    Returns
    -------
    tuple[NDArray[np.floating], list[float], list[float]]
        all_lam (ms), list of zeroth-order responses, list of first-order responses.
    """
    all_lam = np.linspace(1e-4, max_lam, n_lam)
    all_e0: list[float] = []
    all_e1: list[float] = []
    for lam in all_lam:
        lam = lam * 1.0e-3

        E0 = get2_eddy_mode0(G, lam, dt)
        all_e0.append(float(np.sum(E0 * G)))

        E1 = get2_eddy_mode1(G, lam, dt)
        all_e1.append(float(np.sum(E1 * G)))

    return all_lam, all_e0, all_e1


def get_min_TE(
    params: MutableMapping[str, Any],
    bval: float = 1000,
    min_TE: float = -1,
    max_TE: float = -1,
    verbose: int = 0,
) -> tuple[NDArray[np.floating], float]:
    """
    Find minimum TE satisfying constraints for diffusion or free-mode sequences.

    Parameters
    ----------
    params : dict
        Sequence parameters; must include 'mode', timing, and related entries.
    bval : float, optional
        Target b-value for diffusion mode, by default 1000.
    min_TE : float, optional
        Minimum TE to consider (ms). Auto-filled if negative.
    max_TE : float, optional
        Maximum TE to consider (ms). Auto-filled if negative.
    verbose : int, optional
        Verbosity flag.

    Returns
    -------
    tuple[NDArray[np.floating], float]
        Gradient waveform `G_out` and selected TE `T_out` (ms).
    """
    if params['mode'][:4] == 'diff':
        if min_TE < 0:
            min_TE = params['T_readout'] + params['T_90'] + params['T_180'] + 10

        if max_TE < 0:
            max_TE = 200

        G_out, T_out = get_min_TE_diff(params, bval, min_TE, max_TE, verbose)

    elif params['mode'] == 'free':
        if min_TE < 0:
            min_TE = 0.1

        if max_TE < 0:
            max_TE = 5.0

        G_out, T_out = get_min_TE_free(params, min_TE, max_TE, verbose)

    return G_out, T_out


def get_min_TE_diff(
    params: MutableMapping[str, Any],
    target_bval: float,
    min_TE: float,
    max_TE: float,
    verbose: int = 0,
) -> tuple[NDArray[np.floating], float]:
    """
    Binary search TE for diffusion mode to meet or exceed target b-value.

    Parameters
    ----------
    params : dict
        Sequence parameters. Mutated in-place for 'TE'.
    target_bval : float
        Desired b-value.
    min_TE : float
        Lower bound for TE (ms).
    max_TE : float
        Upper bound for TE (ms).
    verbose : int, optional
        Verbosity flag.

    Returns
    -------
    tuple[NDArray[np.floating], float]
        Gradient waveform `G_out` and selected TE `T_out` (ms).
    """
    T_lo = min_TE
    T_hi = max_TE
    T_range = T_hi - T_lo

    best_time = 999999.9

    if 'dt' in params:
        dt: float = params['dt']
    else:
        dt = 1.0e-3 / params['N0']

    if verbose:
        print('Testing TE =', end='', flush=True)
    while (T_range * 1e-3) > (dt / 4.0):
        params['TE'] = T_lo + (T_range) / 2.0
        if verbose:
            print(' %.3f' % params['TE'], end='', flush=True)
        G, ddebug = gropt.gropt(params)
        ddebug[14]
        bval = get_bval(G, params)
        if bval > target_bval:
            T_hi = params['TE']
            if T_hi < best_time:
                G_out = G
                T_out = T_hi
                best_time = T_hi
        else:
            T_lo = params['TE']
        T_range = T_hi - T_lo

    if verbose:
        print(' Final TE = %.3f ms' % T_out)

    params['TE'] = T_out
    return G_out, T_out


def get_min_TE_free(
    params: MutableMapping[str, Any],
    min_TE: float,
    max_TE: float,
    verbose: int = 0,
) -> tuple[NDArray[np.floating], float]:
    """
    Binary search TE for free-mode sequences subject to feasibility constraints.

    Parameters
    ----------
    params : dict
        Sequence parameters. Mutated in-place for 'TE'.
    min_TE : float
        Lower bound for TE (ms).
    max_TE : float
        Upper bound for TE (ms).
    verbose : int, optional
        Verbosity flag.

    Returns
    -------
    tuple[NDArray[np.floating], float]
        Gradient waveform `G_out` and selected TE `T_out` (ms).
    """
    T_lo = min_TE
    T_hi = max_TE
    T_range = T_hi - T_lo

    best_time = 999999.9

    if 'dt' in params:
        dt: float = params['dt']
    else:
        dt = 1.0e-3 / params['N0']

    if verbose:
        print('Testing TE =', end='', flush=True)
    while (T_range * 1e-3) > (dt / 4.0):
        params['TE'] = T_lo + (T_range) / 2.0
        if verbose:
            print(' %.3f' % params['TE'], end='', flush=True)
        G, ddebug = gropt.gropt(params)
        lim_break = ddebug[14]
        if lim_break == 0:
            T_hi = params['TE']
            if T_hi < best_time:
                G_out = G
                T_out = T_hi
                best_time = T_hi
        else:
            T_lo = params['TE']
        T_range = T_hi - T_lo

    if verbose:
        print(' Final TE = %.3f ms' % T_out)

    return G_out, T_out


def get_stim(G: NDArray[np.floating], dt: float) -> NDArray[np.floating]:
    """
    Compute peripheral nerve stimulation (PNS) metric for a gradient waveform.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform array of shape (n_axes, n_samples).
    dt : float
        Sample spacing in seconds.

    Returns
    -------
    NDArray[np.floating]
        PNS stimulus over time (length n_samples - 1).
    """
    c = 334e-6
    Smin = 60
    coeff = []
    for i in range(G.shape[1]):
        coeff.append(c / ((c + dt * (G.shape[1] - 1) - dt * i) ** 2.0) / Smin)
    coeff = np.array(coeff)

    stim_all = []
    for ia in range(G.shape[0]):
        stim_out = []
        for j in range(G.shape[1] - 1):
            ss = 0
            for i in range(j + 1):
                ss += coeff[coeff.size - 1 - j + i] * (G[ia, i + 1] - G[ia, i])
            stim_out.append(ss)
        stim_all.append(np.array(stim_out))

    stim_all = np.array(stim_all)
    stim_all = np.sqrt((stim_all**2.0).sum(0))
    return stim_all


def get_moments(
    G: NDArray[np.floating],
    T_readout: float,
    dt: float,
    diffmode: int = 0,
) -> list[float]:
    """
    Compute gradient moments (M0..M4) for a single-axis waveform.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform (1D).
    T_readout : float
        Readout duration in milliseconds.
    dt : float
        Sample spacing in seconds.
    diffmode : int, optional
        If >0, apply sign inversion at TE/2 to simulate diffusion refocusing.

    Returns
    -------
    list[float]
        Moments M0..M4 scaled to micro-units (see code for scaling factors).
    """
    TE = G.size * dt * 1e3 + T_readout
    tINV = int(np.floor(TE / dt / 1.0e3 / 2.0))
    INV = np.ones(G.size)
    if diffmode > 0:
        INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size) * dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = (1e3 * tvec) ** mm

    np.abs(dt * tMat @ (G * INV))
    mm = dt * tMat * (G * INV)[np.newaxis, :]

    out = []
    for i in range(Nm):
        mmt = np.sum(mm[i])
        out.append(float(mmt * 1e6))

    return out


def get_bval(G: NDArray[np.floating], params: MutableMapping[str, Any]) -> float:
    """
    Compute b-value for a (currently single-axis) diffusion gradient waveform.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform array of shape (n_axes, n_samples). Only first axis used.
    params : dict
        Must contain 'TE', 'T_readout', and 'dt'. TE and T_readout in ms.

    Returns
    -------
    float
        b-value in s/mm^2.
    """
    G = G[0]  # TODO: 3-axis case, right now just assumes 1 axis

    TE = params['TE']
    params['T_readout']
    dt = params['dt']

    tINV = int(np.floor(TE / dt / 1.0e3 / 2.0))
    GAMMA = 42.58e3

    INV = np.ones(G.size)
    INV[tINV:] = -1

    Gt = 0.0
    bval = 0.0
    for i in range(G.size):
        if i < tINV:
            Gt += G[i] * dt
        else:
            Gt -= G[i] * dt
        bval += Gt * Gt * dt

    bval *= (GAMMA * 2 * np.pi) ** 2

    return float(bval)


def plot_moments(G: NDArray[np.floating], T_readout: float, dt: float) -> None:
    """
    Plot cumulative normalized gradient moments (M0..M2) for diagnostics.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform (1D).
    T_readout : float
        Readout duration in milliseconds.
    dt : float
        Sample spacing in seconds.
    """
    TE = G.size * dt * 1e3 + T_readout
    tINV = int(np.floor(TE / dt / 1.0e3 / 2.0))
    GAMMA = 42.58e3
    INV = np.ones(G.size)
    INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size) * dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    np.abs(GAMMA * dt * tMat @ (G * INV))
    mm = GAMMA * dt * tMat * (G * INV)[np.newaxis, :]

    plt.figure()
    mmt = np.cumsum(mm[0])
    plt.plot(mmt / np.abs(mmt).max())
    mmt = np.cumsum(mm[1])
    plt.plot(mmt / np.abs(mmt).max())
    mmt = np.cumsum(mm[2])
    plt.plot(mmt / np.abs(mmt).max())
    plt.axhline(0, color='k')


def get_moment_plots(
    G: NDArray[np.floating],
    T_readout: float,
    dt: float,
    diffmode: int = 1,
) -> list[NDArray[np.floating]]:
    """
    Return cumulative gradient moments (M0..M4) for plotting.

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform array of shape (n_axes, n_samples). First axis used.
    T_readout : float
        Readout duration in milliseconds.
    dt : float
        Sample spacing in seconds.
    diffmode : int, optional
        If >0, apply sign inversion at TE/2.

    Returns
    -------
    list[NDArray[np.floating]]
        Cumulative sums for moments M0..M4 (each length n_samples).
    """
    G = G[0]  # TODO: 3-axis case, right now just assumes 1 axis

    TE = G.size * dt * 1e3 + T_readout
    tINV = int(np.floor(TE / dt / 1.0e3 / 2.0))
    INV = np.ones(G.size)
    if diffmode > 0:
        INV[tINV:] = -1
    Nm = 5
    tvec = np.arange(G.size) * dt
    tMat = np.zeros((Nm, G.size))
    for mm in range(Nm):
        tMat[mm] = tvec**mm

    np.abs(dt * tMat @ (G * INV))
    mm = dt * tMat * (G * INV)[np.newaxis, :]

    out: list[NDArray[np.floating]] = []
    for i in range(Nm):
        mmt = np.cumsum(mm[i])
        out.append(mmt)

    return out


def plot_waveform(
    G: NDArray[np.floating],
    params: MutableMapping[str, Any],
    plot_moments: bool = True,
    plot_eddy: bool = True,
    plot_pns: bool = True,
    plot_slew: bool = True,
    suptitle: str = '',
    eddy_lines: Sequence[float] | None = None,
    eddy_range: Sequence[float] = (1e-3, 120, 1000),
) -> None:
    """
    Plot gradient waveform and derived diagnostics (slew, moments, eddy, PNS).

    Parameters
    ----------
    G : NDArray[np.floating]
        Gradient waveform array of shape (n_axes, n_samples).
    params : dict
        Sequence parameters; must include TE, T_readout, mode, and optionally Naxis.
    plot_moments : bool, optional
        Plot gradient moments, by default True.
    plot_eddy : bool, optional
        Plot eddy current response, by default True.
    plot_pns : bool, optional
        Plot PNS stimulus, by default True.
    plot_slew : bool, optional
        Plot slew rate, by default True.
    suptitle : str, optional
        Figure super-title prefix.
    eddy_lines : Sequence[float] | None, optional
        Vertical lines to annotate on eddy plot (ms).
    eddy_range : Sequence[float], optional
        [lam_min_ms, lam_max_ms, n_samples] for eddy sweep.
    """
    if eddy_lines is None:
        eddy_lines = []
    sns.set()
    sns.set_context('talk')

    Naxis = params.get('Naxis', 1)

    TE = params['TE']
    T_readout = params['T_readout']
    diffmode = 0
    if params['mode'][:4] == 'diff':
        diffmode = 1

    dt = (TE - T_readout) * 1.0e-3 / G.shape[1]
    tt = np.arange(G.shape[1]) * dt * 1e3
    tINV = TE / 2.0

    N_plots = 1
    if plot_moments:
        N_plots += 1
    if plot_eddy:
        N_plots += 1
    if plot_pns:
        N_plots += 1
    if plot_slew:
        N_plots += 1

    N_rows = 1 + (N_plots - 1) // 3
    N_cols = ceil(N_plots / N_rows)

    f, axarr = plt.subplots(N_rows, N_cols, squeeze=False, figsize=(12, N_rows * 3.5))

    i_row = 0
    i_col = 0

    bval = get_bval(G, params)
    blabel = '    b-value = %.0f $mm^{2}/s$' % bval
    if suptitle:
        f.suptitle(suptitle + blabel)
    elif diffmode > 0:
        f.suptitle(blabel)

    if diffmode > 1:
        axarr[i_row, i_col].axvline(tINV, linestyle='--', color='0.7')

    for ia in range(Naxis):
        axarr[i_row, i_col].plot(tt, G[ia] * 1000)
    axarr[i_row, i_col].set_title('Gradient')
    axarr[i_row, i_col].set_xlabel('t [ms]')
    i_col += 1
    if i_col >= N_cols:
        i_col = 0
        i_row += 1

    if plot_slew:
        for ia in range(Naxis):
            axarr[i_row, i_col].plot(tt[:-1], np.diff(G[ia]) / dt)
        axarr[i_row, i_col].set_title('Slew')
        axarr[i_row, i_col].set_xlabel('t [ms]')

        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_moments:
        mm = get_moment_plots(G, T_readout, dt, diffmode)
        for i in range(3):
            if diffmode == 1:
                mmt = mm[i] / np.abs(mm[i]).max()
            if diffmode == 0:
                if i == 0:
                    mmt = mm[i] * 1e6
                if i == 1:
                    mmt = mm[i] * 1e9
                if i == 2:
                    mmt = mm[i] * 1e12
            axarr[i_row, i_col].plot(tt, mmt)
        axarr[i_row, i_col].set_title('Moment [mT/m x $ms^{n}$]')
        if diffmode == 1:
            axarr[i_row, i_col].set_title('Moment [A.U.]')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        axarr[i_row, i_col].legend(
            ('$M_{0}$', '$M_{1}$', '$M_{2}$'),
            prop={'size': 12},
            labelspacing=-0.1,
            loc=0,
        )
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_eddy:
        all_lam = np.linspace(eddy_range[0], eddy_range[1], int(eddy_range[2]))
        all_e = []
        for lam in all_lam:
            lam = lam * 1.0e-3
            r = np.diff(np.exp(-np.arange(G[0].size + 1) * dt / lam))[::-1]
            all_e.append(float(100 * r @ G[0]))

        for e in eddy_lines:
            axarr[i_row, i_col].axvline(e, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))

        axarr[i_row, i_col].axhline(linestyle='--', color='0.7')
        axarr[i_row, i_col].plot(all_lam, all_e)
        axarr[i_row, i_col].set_title('Eddy')
        axarr[i_row, i_col].set_xlabel('\\lambda [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    if plot_pns:
        pns = np.abs(get_stim(G, dt))

        axarr[i_row, i_col].axhline(1.0, linestyle=':', color=(0.8, 0.1, 0.1, 0.8))

        axarr[i_row, i_col].plot(tt[:-1], pns)
        axarr[i_row, i_col].set_title('PNS')
        axarr[i_row, i_col].set_xlabel('Time [ms]')
        i_col += 1
        if i_col >= N_cols:
            i_col = 0
            i_row += 1

    plt.tight_layout(w_pad=0.0, rect=[0, 0.03, 1, 0.95])


def conventional_flowcomp(
    params: MutableMapping[str, Any],
) -> tuple[NDArray[np.floating], float, float, float, float, NDArray[np.floating]]:
    """
    Build a conventional flow-compensation (FC) gradient and its moments.

    Parameters
    ----------
    params : dict
        Must include 'g_ss', 'p_ss', 'dt', 'smax'.

    Returns
    -------
    tuple
        (FC waveform, M0S, M1S, M2S, t_ss, G_ss)
    """
    G_ss = np.concatenate(
        (
            np.linspace(
                params['g_ss'] * 1e-3,
                params['g_ss'] * 1e-3,
                ceil(params['p_ss'] * 1e-3 / params['dt']),
            ),
            np.linspace(params['g_ss'] * 1e-3, params['g_ss'] * 1e-3 / 10, 10),
        ),
        axis=0,
    )
    t_ss = G_ss.size * params['dt']
    mmt = get_moment_plots(G_ss, 0, params['dt'], 0)
    M0S = float(mmt[0][-1] * 1e6)
    M1S = float(mmt[1][-1] * 1e9)
    M2S = float(mmt[2][-1] * 1e12)

    ramp_range = np.linspace(0.01, 0.5, 491)
    for r in ramp_range:
        params['h'] = r * params['smax']
        M02 = (
            -params['h'] * r
            + np.sqrt(
                (params['h'] * r) ** 2
                + 2 * (params['h'] * r * M0S + M0S**2 + 2 * params['h'] * M1S)
            )
        ) / 2
        M01 = M02 + M0S
        w1 = M01 / params['h'] + r
        w2 = M02 / params['h'] + r
        r_ = ceil(r / params['dt'] / 1000)
        w1_ = ceil(w1 / params['dt'] / 1000)
        w2_ = ceil(w2 / params['dt'] / 1000)

        if (w1_ - 2 * r_) == 0 or (w2_ - 2 * r_) == 0:
            break

    G = np.concatenate(
        (
            np.linspace(0, -params['h'], r_),
            np.linspace(-params['h'], -params['h'], w1_ - 2 * r_),
            np.linspace(-params['h'], params['h'], 2 * r_),
            np.linspace(params['h'], params['h'], w2_ - 2 * r_),
            np.linspace(params['h'], 0, r_),
        ),
        axis=0,
    )
    FC = np.concatenate((G_ss * 1000, G), axis=0)

    return FC, M0S, M1S, M2S, t_ss, G_ss


def conventional_flowencode(
    params: MutableMapping[str, Any],
) -> tuple[NDArray[np.floating], float]:
    """
    Build a conventional flow-encoding (FE) gradient given precomputed FC stats.

    Parameters
    ----------
    params : dict
        Must include 'smax', 'VENC', 'M0S', 'G_ss', and 'dt'.

    Returns
    -------
    tuple
        (FE waveform, DeltaM1)
    """
    GAM = 2 * np.pi * 42.57
    DeltaM1 = np.pi / (GAM * params['VENC'])

    ramp_range = np.linspace(0.01, 0.5, 491)
    for r in ramp_range:
        params['h'] = r * params['smax']
        M02 = (
            -params['h'] * r
            + np.sqrt(
                (params['h'] * r) ** 2
                + 2
                * (
                    params['h'] * r * params['M0S']
                    + params['M0S'] ** 2
                    + 2 * params['h'] * (params['M1S'] + DeltaM1)
                )
            )
        ) / 2
        M01 = params['M0S'] + M02
        w1 = M01 / params['h'] + r
        w2 = M02 / params['h'] + r
        r_ = ceil(r / params['dt'] / 1000)
        w1_ = ceil(w1 / params['dt'] / 1000)
        w2_ = ceil(w2 / params['dt'] / 1000)

        if (w1_ == 2 * r_) or (w2_ == 2 * r_):
            break

    G = np.concatenate(
        (
            np.linspace(0, -params['h'], r_),
            np.linspace(-params['h'], -params['h'], w1_ - 2 * r_),
            np.linspace(-params['h'], 0, r_),
            np.linspace(params['h'] / r_, params['h'], r_ - 1),
            np.linspace(params['h'], params['h'], w2_ - 2 * r_),
            np.linspace(params['h'], 0, r_),
        ),
        axis=0,
    )
    FE = np.concatenate((params['G_ss'] * 1000, G), axis=0)

    return FE, float(DeltaM1)


def monopolar_diffusion(
    params: MutableMapping[str, Any],
) -> tuple[NDArray[np.floating], int, float, MutableMapping[str, Any]]:
    """
    Synthesize a monopolar diffusion gradient waveform meeting a target b-value.

    Parameters
    ----------
    params : dict
        Must include 'gmax', 'smax', 'T_90', 'T_180', 'T_readout', 'b'. Values
        are provided in scanner units (ms, mT/m) and converted internally.

    Returns
    -------
    tuple
        (Mono waveform, TE samples, achieved b-value, updated params)
    """
    params['dt'] = 1e-5
    params['T_90'] = params['T_90'] * 1e-3
    params['T_180'] = params['T_180'] * 1e-3
    params['T_readout'] = params['T_readout'] * 1e-3
    h = params['gmax'] / 1000
    SR_Max = params['smax'] / 1000
    GAM = 2 * np.pi * 42.58e3
    zeta = (h / SR_Max) * 1e-3
    Delta = (
        (
            zeta**2 / 12
            + (
                GAM**2 * params['T_180'] * h**2
                - GAM**2 * params['T_90'] * h**2
                + GAM**2 * params['T_readout'] * h**2
                + GAM**2 * h**2 * zeta
            )
            ** 2
            / (4 * GAM**4 * h**4)
        )
        / (
            -(
                (
                    zeta**2 / 12
                    + (
                        GAM**2 * params['T_180'] * h**2
                        - GAM**2 * params['T_90'] * h**2
                        + GAM**2 * params['T_readout'] * h**2
                        + GAM**2 * h**2 * zeta
                    )
                    ** 2
                    / (4 * GAM**4 * h**4)
                )
                ** 3
            )
            + (
                (
                    (
                        GAM**2 * params['T_180'] * h**2
                        - GAM**2 * params['T_90'] * h**2
                        + GAM**2 * params['T_readout'] * h**2
                        + GAM**2 * h**2 * zeta
                    )
                    ** 3
                    / (8 * GAM**6 * h**6)
                    - (
                        -3 * params['b']
                        - GAM**2 * params['T_90'] ** 3 * h**2
                        + GAM**2 * params['T_180'] ** 3 * h**2
                        + GAM**2 * params['T_readout'] ** 3 * h**2
                        + (8 * GAM**2 * h**2 * zeta**3) / 5
                        - 3 * GAM**2 * params['T_90'] * params['T_180'] ** 2 * h**2
                        + 3 * GAM**2 * params['T_90'] ** 2 * params['T_180'] * h**2
                        - 3 * GAM**2 * params['T_90'] * params['T_readout'] ** 2 * h**2
                        + 3 * GAM**2 * params['T_90'] ** 2 * params['T_readout'] * h**2
                        + 3 * GAM**2 * params['T_180'] * params['T_readout'] ** 2 * h**2
                        + 3 * GAM**2 * params['T_180'] ** 2 * params['T_readout'] * h**2
                        - (7 * GAM**2 * params['T_90'] * h**2 * zeta**2) / 2
                        + 3 * GAM**2 * params['T_90'] ** 2 * h**2 * zeta
                        + (7 * GAM**2 * params['T_180'] * h**2 * zeta**2) / 2
                        + 3 * GAM**2 * params['T_180'] ** 2 * h**2 * zeta
                        + (7 * GAM**2 * params['T_readout'] * h**2 * zeta**2) / 2
                        + 3 * GAM**2 * params['T_readout'] ** 2 * h**2 * zeta
                        - 6
                        * GAM**2
                        * params['T_90']
                        * params['T_180']
                        * params['T_readout']
                        * h**2
                        - 6 * GAM**2 * params['T_90'] * params['T_180'] * h**2 * zeta
                        - 6
                        * GAM**2
                        * params['T_90']
                        * params['T_readout']
                        * h**2
                        * zeta
                        + 6
                        * GAM**2
                        * params['T_180']
                        * params['T_readout']
                        * h**2
                        * zeta
                    )
                    / (4 * GAM**2 * h**2)
                    + (
                        zeta**2
                        * (
                            GAM**2 * params['T_180'] * h**2
                            - GAM**2 * params['T_90'] * h**2
                            + GAM**2 * params['T_readout'] * h**2
                            + GAM**2 * h**2 * zeta
                        )
                    )
                    / (16 * GAM**2 * h**2)
                )
                ** 2
            )
            ** (1 / 2)
            + (
                GAM**2 * params['T_180'] * h**2
                - GAM**2 * params['T_90'] * h**2
                + GAM**2 * params['T_readout'] * h**2
                + GAM**2 * h**2 * zeta
            )
            ** 3
            / (8 * GAM**6 * h**6)
            - (
                3
                * (
                    -(GAM**2) * params['T_90'] ** 3 * h**2 / 3
                    + GAM**2 * params['T_90'] ** 2 * params['T_180'] * h**2
                    + GAM**2 * params['T_90'] ** 2 * params['T_readout'] * h**2
                    + GAM**2 * params['T_90'] ** 2 * h**2 * zeta
                    - GAM**2 * params['T_90'] * params['T_180'] ** 2 * h**2
                    - 2
                    * GAM**2
                    * params['T_90']
                    * params['T_180']
                    * params['T_readout']
                    * h**2
                    - 2 * GAM**2 * params['T_90'] * params['T_180'] * h**2 * zeta
                    - GAM**2 * params['T_90'] * params['T_readout'] ** 2 * h**2
                    - 2 * GAM**2 * params['T_90'] * params['T_readout'] * h**2 * zeta
                    - (7 * GAM**2 * params['T_90'] * h**2 * zeta**2) / 6
                    + GAM**2 * params['T_180'] ** 3 * h**2 / 3
                    + GAM**2 * params['T_180'] ** 2 * params['T_readout'] * h**2
                    + GAM**2 * params['T_180'] ** 2 * h**2 * zeta
                    + GAM**2 * params['T_180'] * params['T_readout'] ** 2 * h**2
                    + 2 * GAM**2 * params['T_180'] * params['T_readout'] * h**2 * zeta
                    + (7 * GAM**2 * params['T_180'] * h**2 * zeta**2) / 6
                    + GAM**2 * params['T_readout'] ** 3 * h**2 / 3
                    + GAM**2 * params['T_readout'] ** 2 * h**2 * zeta
                    + (7 * GAM**2 * params['T_readout'] * h**2 * zeta**2) / 6
                    + (8 * GAM**2 * h**2 * zeta**3) / 15
                    - params['b']
                )
            )
            / (4 * GAM**2 * h**2)
            + (
                zeta**2
                * (
                    GAM**2 * params['T_180'] * h**2
                    - GAM**2 * params['T_90'] * h**2
                    + GAM**2 * params['T_readout'] * h**2
                    + GAM**2 * h**2 * zeta
                )
            )
            / (16 * GAM**2 * h**2)
        )
        ** (1 / 3)
        + (
            (
                (
                    GAM**2 * params['T_180'] * h**2
                    - GAM**2 * params['T_90'] * h**2
                    + GAM**2 * params['T_readout'] * h**2
                    + GAM**2 * h**2 * zeta
                )
                ** 3
            )
            / (8 * GAM**6 * h**6)
            - (
                3
                * (
                    -(GAM**2) * params['T_90'] ** 3 * h**2 / 3
                    + GAM**2 * params['T_90'] ** 2 * params['T_180'] * h**2
                    + GAM**2 * params['T_90'] ** 2 * params['T_readout'] * h**2
                    + GAM**2 * params['T_90'] ** 2 * h**2 * zeta
                    - GAM**2 * params['T_90'] * params['T_180'] ** 2 * h**2
                    - 2
                    * GAM**2
                    * params['T_90']
                    * params['T_180']
                    * params['T_readout']
                    * h**2
                    - 2 * GAM**2 * params['T_90'] * params['T_180'] * h**2 * zeta
                    - GAM**2 * params['T_90'] * params['T_readout'] ** 2 * h**2
                    - 2 * GAM**2 * params['T_90'] * params['T_readout'] * h**2 * zeta
                    - (7 * GAM**2 * params['T_90'] * h**2 * zeta**2) / 6
                    + GAM**2 * params['T_180'] ** 3 * h**2 / 3
                    + GAM**2 * params['T_180'] ** 2 * params['T_readout'] * h**2
                    + GAM**2 * params['T_180'] ** 2 * h**2 * zeta
                    + GAM**2 * params['T_180'] * params['T_readout'] ** 2 * h**2
                    + 2 * GAM**2 * params['T_180'] * params['T_readout'] * h**2 * zeta
                    + (7 * GAM**2 * params['T_180'] * h**2 * zeta**2) / 6
                    + GAM**2 * params['T_readout'] ** 3 * h**2 / 3
                    + GAM**2 * params['T_readout'] ** 2 * h**2 * zeta
                    + (7 * GAM**2 * params['T_readout'] * h**2 * zeta**2) / 6
                    + (8 * GAM**2 * h**2 * zeta**3) / 15
                    - params['b']
                )
            )
            / (4 * GAM**2 * h**2)
            + (
                zeta**2
                * (
                    GAM**2 * params['T_180'] * h**2
                    - GAM**2 * params['T_90'] * h**2
                    + GAM**2 * params['T_readout'] * h**2
                    + GAM**2 * h**2 * zeta
                )
            )
            / (16 * GAM**2 * h**2)
        )
        ** (1 / 3)
        + (
            GAM**2 * params['T_180'] * h**2
            - GAM**2 * params['T_90'] * h**2
            + GAM**2 * params['T_readout'] * h**2
            + GAM**2 * h**2 * zeta
        )
        / (2 * GAM**2 * h**2)
    )
    delta = Delta + params['T_90'] - params['T_180'] - params['T_readout'] - zeta
    b = (
        GAM**2
        * h**2
        * (delta**2 * (Delta - delta / 3) + zeta**3 / 30 - delta * zeta**2 / 6)
    )
    T_90_ = int(1e5 * params['T_90'])
    zeta_ = int(1e5 * zeta)
    delta_ = int(1e5 * (delta - 2 * zeta))
    Delta_ = int(1e5 * (Delta - zeta - delta + params['T_180'] / 2))
    Mono = np.concatenate(
        (
            np.linspace(0, 0, T_90_),
            np.linspace(0, h, zeta_),
            np.linspace(h, h, delta_),
            np.linspace(h, 0, zeta_),
            np.linspace(0, 0, Delta_),
            np.linspace(0, h, zeta_),
            np.linspace(h, h, delta_),
            np.linspace(h, 0, zeta_),
        )
    )
    TE = Mono.size

    return Mono, TE, float(b), params


def bipolar_diffusion(
    params: MutableMapping[str, Any],
) -> tuple[NDArray[np.floating], int, float, MutableMapping[str, Any]]:
    """
    Synthesize a bipolar diffusion gradient waveform meeting a target b-value.

    Parameters
    ----------
    params : dict
        Must include 'gmax', 'smax', 'T_90', 'T_180', 'T_readout', 'b'.

    Returns
    -------
    tuple
        (Bipolar waveform, TE samples, achieved b-value, updated params)
    """
    params['dt'] = 1e-5
    params['T_90'] = params['T_90'] * 1e-3
    params['T_180'] = params['T_180'] * 1e-3
    params['T_readout'] = params['T_readout'] * 1e-3
    h = params['gmax'] / 1000
    SR_Max = params['smax'] / 1000
    GAM = 2 * np.pi * 42.58e3
    zeta = (h / SR_Max) * 1e-3
    delta = (
        (
            (9 * (params['b'] - (GAM**2 * h**2 * zeta**3) / 15) ** 2)
            / (64 * GAM**4 * h**4)
            - zeta**6 / 1728
        )
        ** (1 / 2)
        + (3 * (params['b'] - (GAM**2 * h**2 * zeta**3) / 15)) / (8 * GAM**2 * h**2)
    ) ** (1 / 3) + zeta**2 / (
        12
        * (
            (
                (9 * (params['b'] - (GAM**2 * h**2 * zeta**3) / 15) ** 2)
                / (64 * GAM**4 * h**4)
                - zeta**6 / 1728
            )
            ** (1 / 2)
            + (3 * params['b'] - (GAM**2 * h**2 * zeta**3) / 5) / (8 * GAM**2 * h**2)
        )
        ** (1 / 3)
    )
    b = (GAM**2 * h**2 * (20 * delta**3 - 5 * delta * zeta**2 + zeta**3)) / 15
    T_90_ = int(1e5 * params['T_90'])
    T_180_ = int(1e5 * params['T_180'])
    T_readout_ = int(1e5 * params['T_readout'])
    zeta_ = int(1e5 * zeta)
    delta_ = int(1e5 * (delta - 2 * zeta))
    gap = T_readout_ - 0.5 * T_90_
    Bipolar = np.concatenate(
        (
            np.linspace(0, 0, T_90_),
            np.linspace(0, h, zeta_),
            np.linspace(h, h, delta_),
            np.linspace(h, 0, zeta_),
            np.linspace(0, -h, zeta_),
            np.linspace(-h, -h, delta_),
            np.linspace(-h, 0, zeta_),
            np.linspace(0, 0, gap),
            np.linspace(0, 0, T_180_),
            np.linspace(0, h, zeta_),
            np.linspace(h, h, delta_),
            np.linspace(h, 0, zeta_),
            np.linspace(0, -h, zeta_),
            np.linspace(-h, -h, delta_),
            np.linspace(-h, 0, zeta_),
        )
    )
    TE = Bipolar.size

    return Bipolar, TE, float(b), params


def asymmbipolar_diffusion(
    params: MutableMapping[str, Any],
) -> tuple[NDArray[np.floating], int, float, MutableMapping[str, Any]]:
    """
    Synthesize an asymmetric bipolar diffusion gradient (fixed deltas).

    Parameters
    ----------
    params : dict
        Must include 'gmax', 'smax', 'T_90', 'T_180', 'T_readout'. Uses fixed
        delta1, delta2, Delta values (see code).

    Returns
    -------
    tuple
        (Asymmetric bipolar waveform, TE samples, b-value, updated params)
    """
    params['dt'] = 1e-5
    params['T_90'] = params['T_90'] * 1e-3
    params['T_180'] = params['T_180'] * 1e-3
    params['T_readout'] = params['T_readout'] * 1e-3
    h = params['gmax'] / 1000
    SR_Max = params['smax'] / 1000
    GAM = 2 * np.pi * 42.58e3
    zeta = (h / SR_Max) * 1e-3
    delta1 = 0.0135
    delta2 = 0.0210
    Delta = 0.0733
    b = (
        GAM**2
        * h**2
        * (
            20 * Delta**3 * delta1**3
            - 30 * Delta**3 * delta1**2 * zeta
            - 5 * Delta**3 * delta1 * zeta**2
            + 16 * Delta**3 * zeta**3
            - 60 * Delta**2 * delta1**4
            + 120 * Delta**2 * delta1**3 * zeta
            - 5 * Delta**2 * delta1**2 * zeta**2
            - 106 * Delta**2 * delta1 * zeta**3
            + 48 * Delta**2 * zeta**4
            - 100 * Delta * delta1**3 * zeta**2
            + 252 * Delta * delta1**2 * zeta**3
            - 197 * Delta * delta1 * zeta**4
            + 48 * Delta * zeta**5
            + 40 * delta1**6
            - 120 * delta1**5 * zeta
            + 200 * delta1**4 * zeta**2
            - 268 * delta1**3 * zeta**3
            + 227 * delta1**2 * zeta**4
            - 96 * delta1 * zeta**5
            + 16 * zeta**6
        )
        / (15 * (Delta - 2 * delta1 + zeta) ** 3)
    )
    T_90_ = int(1e5 * params['T_90'])
    int(1e5 * params['T_180'])
    int(1e5 * params['T_readout'])
    zeta_ = int(1e5 * zeta)
    delta1_ = int(1e5 * (delta1 - 2 * zeta))
    delta2_ = int(1e5 * (delta2 - 2 * zeta))
    gap = int(1e5 * (Delta - delta1 - delta2))
    AsymmBipolar = np.concatenate(
        (
            np.linspace(0, 0, T_90_),
            np.linspace(0, -h, zeta_),
            np.linspace(-h, -h, delta1_),
            np.linspace(-h, 0, zeta_),
            np.linspace(0, h, zeta_),
            np.linspace(h, h, delta2_),
            np.linspace(h, 0, zeta_),
            np.linspace(0, 0, gap),
            np.linspace(0, h, zeta_),
            np.linspace(h, h, delta2_),
            np.linspace(h, 0, zeta_),
            np.linspace(0, -h, zeta_),
            np.linspace(-h, -h, delta1_),
            np.linspace(-h, 0, zeta_),
        ),
        axis=0,
    )
    TE = AsymmBipolar.size

    return AsymmBipolar, TE, float(b), params
