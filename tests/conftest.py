import pytest

from tests.synthetic_conll import make_data_root


@pytest.fixture()
def data_root(tmp_path):
    """Folder with a synthetic train and dev file in the real CoNLL format (no test file)."""
    return make_data_root(tmp_path)
