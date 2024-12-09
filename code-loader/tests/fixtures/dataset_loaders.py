import pytest

from code_loader.leaploader import LeapLoader
from tests.fixtures.dataset_integ_scripts import dataset_integ_scripts_path

@pytest.fixture
def no_cloud_input_dim_dataset_loader() -> LeapLoader:
    file_name = "simple_no_cloud_dataset_channel_dim.py"
    leap_loader = LeapLoader(dataset_integ_scripts_path, file_name)
    return leap_loader


@pytest.fixture
def no_cloud_dataset_loader() -> LeapLoader:
    file_name = "simple_no_cloud_dataset.py"
    leap_loader = LeapLoader(dataset_integ_scripts_path, file_name)
    return leap_loader


@pytest.fixture
def no_cloud_wt_visualizer_dataset_loader() -> LeapLoader:
    file_name = "simple_no_cloud_dataset_wt_visualizer.py"
    leap_loader = LeapLoader(dataset_integ_scripts_path, file_name)
    return leap_loader


@pytest.fixture
def secret_dataset_loader() -> LeapLoader:
    file_name = "simple_dataset_wt_secret.py"
    leap_loader = LeapLoader(dataset_integ_scripts_path, file_name)
    return leap_loader


@pytest.fixture
def word_idx_dataset_loader() -> LeapLoader:
    file_name = "simple_no_cloud_dataset_wt_word_idx.py"
    leap_loader = LeapLoader(dataset_integ_scripts_path, file_name)
    return leap_loader