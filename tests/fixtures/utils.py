import pytest

from code_loader import dataset_binder
from code_loader.contract.datasetclasses import DatasetIntegrationSetup


def use_fixture(fixture_func):
    def inner(test_func):
        return pytest.mark.usefixtures(fixture_func.__name__)(test_func)

    return inner


@pytest.fixture
def refresh_setup_container() -> None:
    dataset_binder.setup_container = DatasetIntegrationSetup()
