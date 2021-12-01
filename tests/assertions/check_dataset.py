from typing import List, Optional

from code_loader.contract.responsedataclasses import DatasetIntegParseResult, DatasetSetup, DatasetTestResultPayload
from grappa import should  # type: ignore


def assert_setup_is_valid(actual_setup: Optional[DatasetSetup], expected_setup: Optional[DatasetSetup]) -> None:
    actual_setup | should.be.equal(expected_setup)


def assert_payload_is_valid(actual_payloads: List[DatasetTestResultPayload],
                            expected_payloads: List[DatasetTestResultPayload]) -> None:
    actual_payloads | should.be.equal(expected_payloads)


def assert_dataset_integ_parse_result_is_valid(parse_result: DatasetIntegParseResult,
                                               expected_result: DatasetIntegParseResult) -> None:
    parse_result.is_valid | should.be.equal(expected_result.is_valid)
    parse_result.general_error | should.be.equal(expected_result.general_error)

    assert_setup_is_valid(parse_result.setup, expected_result.setup)
    assert_payload_is_valid(parse_result.payloads, expected_result.payloads)
    # for payload in parse_result.payloads:
    #     assert_payload_is_valid(payload)
