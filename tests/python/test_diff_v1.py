import gropt

from .helpers import compare_waveforms, load_case, normalize_params


def test_diff_v1(casefile):
    data = load_case(casefile)
    version = data.get("version", None)
    assert version == "diff_v1", f"unexpected case version {version} in {casefile}"

    params = normalize_params(data)
    G_expected = data["G"]

    G_actual, _limit_break = gropt.gropt(params)
    compare_waveforms(G_expected, G_actual)
