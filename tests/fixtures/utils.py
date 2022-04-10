from typing import Union, Dict

import pytest

from code_loader import dataset_binder
from code_loader.contract.enums import DataStateEnum
from code_loader.dataset_binder import DatasetBinder


def use_fixture(fixture_func):
    def inner(test_func):
        return pytest.mark.usefixtures(fixture_func.__name__)(test_func)

    return inner


@pytest.fixture
def refresh_setup_container() -> None:
    new_dataset_binder = DatasetBinder()
    dataset_binder.setup_container = new_dataset_binder.setup_container
    dataset_binder.cache_container = new_dataset_binder.cache_container


@pytest.fixture
def simple_sample_params() -> Dict[str, Union[DataStateEnum, int]]:
    return {"state": DataStateEnum.training, "idx": 0}
