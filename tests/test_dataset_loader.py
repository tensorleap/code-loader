from tests.assertions import assert_dataset_binder_is_valid, assert_encoder_is_valid, \
    assert_sample_is_valid, assert_secret_exists
from tests.fixtures.dataset_loaders import no_cloud_dataset_loader, secret_dataset_loader
from tests.fixtures.secrets.tensorleap_demo_secret import put_mock_secret_in_env
from tests.fixtures.utils import refresh_setup_container
from tests.fixtures.utils import use_fixture, simple_sample_params


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
def test_exec_script_no_cloud(no_cloud_dataset_loader, refresh_setup_container):
    # act
    no_cloud_dataset_loader.exec_script()

    # assert
    assert_dataset_binder_is_valid()


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(simple_sample_params)
def test_get_inputs_no_cloud(no_cloud_dataset_loader, refresh_setup_container, simple_sample_params):
    # act
    no_cloud_dataset_loader.exec_script()
    inputs = no_cloud_dataset_loader._get_inputs(**simple_sample_params)

    # assert
    assert_dataset_binder_is_valid()
    assert_encoder_is_valid(inputs)


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(simple_sample_params)
def test_get_gt_no_cloud(no_cloud_dataset_loader, refresh_setup_container, simple_sample_params):
    # act
    no_cloud_dataset_loader.exec_script()
    gt = no_cloud_dataset_loader._get_gt(**simple_sample_params)

    # assert
    assert_dataset_binder_is_valid()
    assert_encoder_is_valid(gt)


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(simple_sample_params)
def test_get_metadata_no_cloud(no_cloud_dataset_loader, refresh_setup_container, simple_sample_params):
    # act
    no_cloud_dataset_loader.exec_script()
    metadata = no_cloud_dataset_loader._get_metadata(**simple_sample_params)

    # assert
    assert_dataset_binder_is_valid()
    assert_encoder_is_valid(metadata)


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(simple_sample_params)
def test_get_sample_dataset_loader_no_cloud(no_cloud_dataset_loader, refresh_setup_container, simple_sample_params):
    # act
    no_cloud_dataset_loader.exec_script()
    sample = no_cloud_dataset_loader.get_sample(**simple_sample_params)

    # assert
    assert_dataset_binder_is_valid()
    assert_sample_is_valid(sample)


@use_fixture(secret_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(put_mock_secret_in_env)
def test_exec_script_secret_is_loaded(secret_dataset_loader, refresh_setup_container, put_mock_secret_in_env):
    # act
    secret_dataset_loader.exec_script()

    # assert
    assert_dataset_binder_is_valid()
    assert_secret_exists(secret_dataset_loader)
