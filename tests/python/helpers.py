import numpy as np
import numpy.testing as npt

import hdf5storage


def _as_str(val):
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return val


def load_case(path):
    return hdf5storage.read(filename=path)


def normalize_params(data):
    params = data["params_in"]
    diffmode = params.get("diffmode", None)
    if diffmode == 1:
        params["mode"] = "diff_beta"
    elif diffmode == 2:
        params["mode"] = "diff_bval"
    if "N" in params and params["N"] is not None and params["N"] > 0:
        params["N0"] = params["N"]
    return params


def compare_waveforms(G_expected, G_actual, rtol=1e-3):
    G_expected = np.asarray(G_expected)
    G_actual = np.asarray(G_actual)

    # If actual is (1, N) and expected is (N,), align to (1, N)
    if G_actual.ndim == 2 and G_actual.shape[0] == 1 and G_expected.ndim == 1:
        G_expected = G_expected.reshape(1, -1)

    # Check
    npt.assert_allclose(G_actual, G_expected, rtol=rtol)
