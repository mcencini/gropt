"""
High-level Python wrapper for GrOpt bindings.

This module mirrors the original Cython API, using the pybind11 extension
``_gropt_binding`` and provides NumPy-friendly helpers.
"""

from __future__ import annotations

__all__ = []

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._gropt_binding import (
    run_kernel_diff_fixeddt,
    run_kernel_diff_fixeddt_fixG,
    run_kernel_diff_fixedN,
    run_kernel_diff_fixedN_Gin,
)

NDArrayF64 = NDArray[np.float64]


def gropt(params: dict[str, Any], verbose: int = 0) -> tuple[NDArrayF64, NDArrayF64]:
    """
    High-level GrOpt API for orchestrating kernel execution.

    Parameters
    ----------
    params : dict
        Optimization parameters. Required keys: 'mode', 'gmax', 'smax', 'TE'.
        One of: 'N0' (fixed steps) or 'dt' (fixed timestep).
        Optional keys: 'moment_params', 'T_90', 'T_180', 'T_readout', 'dt_out',
        'eddy_params', 'pns_thresh', 'gfix', 'slew_reg', 'Naxis'.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    (NDArrayF64, NDArrayF64)
        Optimized gradients (shape (Naxis, -1)) and debug output (len 100).
    """
    required = {'mode', 'gmax', 'smax', 'TE'}
    missing = required - params.keys()
    if missing:
        raise ValueError(f'Missing required keys: {missing}')

    mode_to_diffmode = {'diff_bval': 2, 'diff_beta': 1, 'free': 0}
    mode = params['mode']
    if mode not in mode_to_diffmode:
        raise ValueError(
            f"Invalid mode '{mode}', must be one of {list(mode_to_diffmode)}"
        )
    diffmode = mode_to_diffmode[mode]

    gmax = params['gmax'] / 1000.0 if params['gmax'] > 1.0 else params['gmax']
    smax = params['smax']
    TE = params['TE']
    T_90 = params.get('T_90', 0.0)
    T_180 = params.get('T_180', 0.0)
    T_readout = params.get('T_readout', 0.0)
    dt_out = params.get('dt_out', -1.0)
    slew_reg = params.get('slew_reg', 1.0)
    pns_thresh = params.get('pns_thresh', -1.0)
    Naxis = params.get('Naxis', 1)

    moment_params = array_prep(
        np.array(
            params.get('moment_params', [[0, 0, 0, -1, -1, 0, 1.0e-3]]),
            dtype=np.float64,
        )
    )
    eddy_params = array_prep(np.array(params.get('eddy_params', []), dtype=np.float64))
    gfix = array_prep(np.array(params.get('gfix', []), dtype=np.float64))

    if 'N0' in params:
        N0 = params['N0']
        return run_kernel_diff_fixedN(
            N0,
            gmax,
            smax,
            TE,
            moment_params,
            pns_thresh,
            T_readout,
            T_90,
            T_180,
            diffmode,
            dt_out,
            eddy_params,
            -1.0,
            slew_reg,
            verbose,
        )
    elif 'dt' in params:
        dt = params['dt']
        if gfix.size > 0:
            return run_kernel_diff_fixeddt_fixG(
                dt,
                gmax,
                smax,
                TE,
                moment_params,
                pns_thresh,
                T_readout,
                T_90,
                T_180,
                diffmode,
                dt_out,
                eddy_params,
                -1.0,
                gfix.size,
                gfix,
                slew_reg,
                verbose,
            )
        else:
            return run_kernel_diff_fixeddt(
                dt,
                gmax,
                smax,
                TE,
                moment_params,
                pns_thresh,
                T_readout,
                T_90,
                T_180,
                diffmode,
                dt_out,
                eddy_params,
                -1.0,
                slew_reg,
                Naxis,
                verbose,
            )
    else:
        raise ValueError("params must contain either 'N0' or 'dt'")


# %% Subroutines
def array_prep(array: NDArrayF64, linear: bool = True) -> NDArrayF64:
    """Ensure a NumPy array is C-contiguous, float64, and optionally flattened."""
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)
    array = array.astype(np.float64, copy=False, order='C')
    return array.ravel() if linear else array


def run_diffkernel_fixN(
    gmax: float,
    smax: float,
    MMT: int,
    TE: float,
    T_readout: float,
    T_90: float,
    T_180: float,
    diffmode: int,
    N0: int = 64,
    dt_out: float = -1.0,
    eddy: list[float] | None = None,
    pns_thresh: float = -1.0,
    verbose: int = 1,
) -> tuple[NDArrayF64, NDArrayF64]:
    """Run the kernel with a fixed number of timesteps (N0)."""
    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1e-3])

    return run_kernel_diff_fixedN(
        N0,
        gmax,
        smax,
        TE,
        array_prep(np.array(m_params, dtype=np.float64)),
        pns_thresh,
        T_readout,
        T_90,
        T_180,
        diffmode,
        dt_out,
        array_prep(np.array(eddy or [], dtype=np.float64)),
        -1.0,
        1.0,
        verbose,
    )


def run_diffkernel_fixdt(
    gmax: float,
    smax: float,
    MMT: int,
    TE: float,
    T_readout: float,
    T_90: float,
    T_180: float,
    diffmode: int,
    dt: float = 0.4e-3,
    dt_out: float = -1.0,
    eddy: list[float] | None = None,
    pns_thresh: float = -1.0,
    verbose: int = 1,
) -> tuple[NDArrayF64, NDArrayF64]:
    """Run the kernel with a fixed timestep (dt)."""
    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1e-3])

    return run_kernel_diff_fixeddt(
        dt,
        gmax,
        smax,
        TE,
        array_prep(np.array(m_params, dtype=np.float64)),
        pns_thresh,
        T_readout,
        T_90,
        T_180,
        diffmode,
        dt_out,
        array_prep(np.array(eddy or [], dtype=np.float64)),
        -1.0,
        1.0,
        verbose,
    )


def run_diffkernel_fixN_Gin(
    G_in: NDArrayF64,
    gmax: float,
    smax: float,
    MMT: int,
    TE: float,
    T_readout: float,
    T_90: float,
    T_180: float,
    diffmode: int,
    N0: int = 64,
    dt_out: float = -1.0,
    eddy: list[float] | None = None,
    pns_thresh: float = -1.0,
    verbose: int = 1,
) -> tuple[NDArrayF64, NDArrayF64]:
    """Run the kernel with fixed timesteps (N0) and an input gradient waveform (G_in)."""
    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1e-3]]
    if MMT > 0:
        m_params.append([0.0, 1.0, -1.0, -1.0, 0.0, 1e-3])
    if MMT > 1:
        m_params.append([0.0, 2.0, -1.0, -1.0, 0.0, 1e-3])

    return run_kernel_diff_fixedN_Gin(
        array_prep(np.array(G_in, dtype=np.float64)),
        N0,
        gmax,
        smax,
        TE,
        array_prep(np.array(m_params, dtype=np.float64)),
        pns_thresh,
        T_readout,
        T_90,
        T_180,
        diffmode,
        dt_out,
        array_prep(np.array(eddy or [], dtype=np.float64)),
        -1.0,
        1.0,
        verbose,
    )


def run_diffkernel_fixdt_fixG(
    gmax: float,
    smax: float,
    TE: float,
    T_readout: float,
    T_90: float,
    T_180: float,
    diffmode: int,
    dt: float,
    gfix: NDArrayF64,
    slew_reg: float = 1.0,
    eddy: list[float] | None = None,
    dt_out: float = -1.0,
    pns_thresh: float = -1.0,
    verbose: int = 0,
) -> tuple[NDArrayF64, NDArrayF64]:
    """Run the kernel with fixed timestep (dt) and fixed gradient values (gfix)."""
    eddy_params = array_prep(np.array(eddy or [], dtype=np.float64))
    gfix = array_prep(np.array(gfix, dtype=np.float64))

    m_params = [[0.0, 0.0, -1.0, -1.0, 0.0, 1e-3]]
    return run_kernel_diff_fixeddt_fixG(
        dt,
        gmax,
        smax,
        TE,
        array_prep(np.array(m_params, dtype=np.float64)),
        pns_thresh,
        T_readout,
        T_90,
        T_180,
        diffmode,
        dt_out,
        eddy_params,
        -1.0,
        gfix.size,
        gfix,
        slew_reg,
        verbose,
    )
