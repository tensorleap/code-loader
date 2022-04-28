from code_loader.contract.responsedataclasses import DatasetIntegParseResult
from tests.assertions.check_dataset import assert_dataset_integ_parse_result_is_valid
from tests.assertions.dataset_loader import assert_leap_binder_is_valid
from tests.fixtures.dataset_loaders import no_cloud_dataset_loader, no_cloud_wt_decoder_dataset_loader
from tests.fixtures.results.simple_no_cloud_dataset import no_cloud_dataset_loader_expected_result
from tests.fixtures.results.simple_no_cloud_wt_decoder_dataset import no_cloud_wt_decoder_dataset_loader_expected_result
from tests.fixtures.utils import use_fixture, refresh_setup_container


@use_fixture(no_cloud_dataset_loader_expected_result)
@use_fixture(no_cloud_dataset_loader)
@use_fixture(refresh_setup_container)
def test_check_dataset_no_cloud(no_cloud_dataset_loader, refresh_setup_container,
                                no_cloud_dataset_loader_expected_result):
    # act
    parse_result: DatasetIntegParseResult = no_cloud_dataset_loader.check_dataset()

    # assert
    assert_leap_binder_is_valid()
    assert_dataset_integ_parse_result_is_valid(parse_result, no_cloud_dataset_loader_expected_result)


@use_fixture(no_cloud_wt_decoder_dataset_loader_expected_result)
@use_fixture(no_cloud_wt_decoder_dataset_loader)
@use_fixture(refresh_setup_container)
def test_check_dataset_no_cloud_wt_decoder(no_cloud_wt_decoder_dataset_loader, refresh_setup_container,
                                           no_cloud_wt_decoder_dataset_loader_expected_result):
    # act
    parse_result: DatasetIntegParseResult = no_cloud_wt_decoder_dataset_loader.check_dataset()

    # assert
    assert_leap_binder_is_valid()
    assert_dataset_integ_parse_result_is_valid(parse_result, no_cloud_wt_decoder_dataset_loader_expected_result)
