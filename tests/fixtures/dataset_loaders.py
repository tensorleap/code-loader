import os

import pytest

from code_loader.datasetloader import DatasetLoader
from tests.fixtures.dataset_integ_scripts import dataset_integ_scripts_path


@pytest.fixture
def no_cloud_dataset_loader() -> DatasetLoader:
    script_path = os.path.join(dataset_integ_scripts_path, "simple_no_cloud_dataset.py")
    with open(script_path, "r") as f:
        script = f.read()
    dataset_loader = DatasetLoader(script)
    return dataset_loader


@pytest.fixture
def secret_dataset_loader() -> DatasetLoader:
    script_path = os.path.join(dataset_integ_scripts_path, "simple_dataset_wt_secret.py")
    with open(script_path, "r") as f:
        script = f.read()
    dataset_loader = DatasetLoader(script)
    return dataset_loader


@pytest.fixture
def word_idx_dataset_loader() -> DatasetLoader:
    script_path = os.path.join(dataset_integ_scripts_path, "simple_no_cloud_dataset_wt_word_idx.py")
    with open(script_path, "r") as f:
        script = f.read()
    dataset_loader = DatasetLoader(script)
    return dataset_loader
