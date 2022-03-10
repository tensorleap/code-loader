from tests.assertions.dataset_loader import assert_dataset_binder_is_valid, assert_subsets_is_valid, \
    assert_encoder_is_valid, assert_sample_is_valid, assert_word_to_index_in_cache_container, assert_input_has_value, \
    assert_metadata_encoder_is_valid
from tests.fixtures.dataset_integ_scripts.scripts_metadata import word_idx_dataset_params
from tests.fixtures.dataset_loaders import no_cloud_dataset_loader, word_idx_dataset_loader
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
def test_subsets_no_cloud(no_cloud_dataset_loader, refresh_setup_container):
    # act
    no_cloud_dataset_loader.exec_script()
    subsets = no_cloud_dataset_loader._subsets()

    # assert
    assert_dataset_binder_is_valid()
    assert_subsets_is_valid(subsets)


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
    assert_metadata_encoder_is_valid(metadata)


@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(simple_sample_params)
def test_get_sample_dataset_loader_no_cloud(no_cloud_dataset_loader, refresh_setup_container, simple_sample_params):
    # act
    sample = no_cloud_dataset_loader.get_sample(**simple_sample_params)

    # assert
    assert_dataset_binder_is_valid()
    assert_sample_is_valid(sample)


@use_fixture(word_idx_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(word_idx_dataset_params)
def test_word_to_index(word_idx_dataset_loader, refresh_setup_container, word_idx_dataset_params):
    # act
    word_idx_dataset_loader.exec_script()
    word_idx_dataset_loader._subsets()

    # assert
    assert_dataset_binder_is_valid()
    assert_word_to_index_in_cache_container(word_idx_dataset_params["input_name"],
                                            word_idx_dataset_params["word_to_index_value"])


@use_fixture(word_idx_dataset_loader)
@use_fixture(refresh_setup_container)
@use_fixture(simple_sample_params)
@use_fixture(word_idx_dataset_params)
def test_cache_container_in_encoder(word_idx_dataset_loader, refresh_setup_container, simple_sample_params,
                                    word_idx_dataset_params):
    # act
    word_idx_dataset_loader.exec_script()
    inputs = word_idx_dataset_loader._get_inputs(**simple_sample_params)

    # assert
    assert_dataset_binder_is_valid()
    assert_word_to_index_in_cache_container(word_idx_dataset_params["input_name"],
                                            word_idx_dataset_params["word_to_index_value"])
    assert_input_has_value(inputs[word_idx_dataset_params["input_name"]],
                           word_idx_dataset_params["word_to_index_value"])
