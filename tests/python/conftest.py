from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def case_files():
    here = Path(__file__).resolve().parent  # tests/python
    cases_root = here.parent / "cases"  # tests/cases
    return sorted(cases_root.rglob("*.h5"))


def pytest_generate_tests(metafunc):
    if "casefile" in metafunc.fixturenames:
        here = Path(__file__).resolve().parent
        cases_root = here.parent / "cases"
        files = sorted(cases_root.rglob("*.h5"))
        metafunc.parametrize(
            "casefile",
            files,
            ids=lambda p: p.relative_to(cases_root.parent).as_posix(),
        )
