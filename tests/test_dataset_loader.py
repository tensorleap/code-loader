from tests.assertions import assert_dataset_binder_is_valid
from tests.fixtures.dataset_loaders import no_cloud_dataset_loader
from tests.fixtures.utils import use_fixture
from tests.fixtures.utils import refresh_setup_container


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
def test_DatasetLoader_exec_script_no_cloud(no_cloud_dataset_loader, refresh_setup_container):
    # act
    no_cloud_dataset_loader.exec_script()

    # assert
    assert_dataset_binder_is_valid()


def test_subset_function_dataset_loader_no_cloud():
    # act

    # assert
    pass


def test_generate_sample_dataset_loader_no_cloud():
    # act

    # assert
    pass
