import random
from functools import lru_cache
from typing import Dict, List, cast

import numpy as np  # type: ignore

from code_loader.contract.datasetclasses import SubsetResponse, DatasetSample, DatasetBaseHandler, InputHandler, \
    GroundTruthHandler, SubsetHandler
from code_loader.contract.enums import DataStateEnum, TestingSectionEnum, DataStateType
from code_loader.contract.responsedataclasses import DatasetIntegParseResult, DatasetTestResultPayload, \
    DatasetSubsetInstance, DatasetSetup, DatasetInputInstance, DatasetOutputInstance, DatasetMetadataInstance
from code_loader.dataset_binder import global_dataset_binder
from code_loader.utils import get_root_exception_line_number, get_shape


# TODO: add handling of large return messages (usually return of large subsets)
# MAX_PUBSUB_MSG_SIZE = 500000


class DatasetLoader:

    def __init__(self, dataset_script: str):
        self.dataset_script: str = dataset_script
        self.index_dict: Dict[str, int] = {}
        self.global_variables = {'index_dict': self.index_dict}

    @lru_cache()
    def exec_script(self):
        exec(self.dataset_script, self.global_variables)

    def get_sample(self, state: DataStateEnum, idx: int) -> DatasetSample:
        self.exec_script()
        sample = DatasetSample(inputs=self._get_inputs(state, idx),
                               gt=self._get_gt(state, idx),
                               metadata=self._get_metadata(state, idx),
                               index=idx,
                               state=state)
        return sample

    def check_dataset(self) -> DatasetIntegParseResult:
        test_payloads: List[DatasetTestResultPayload] = []
        setup_response = None
        general_error = None
        try:
            self.exec_script()
            subset_test_payloads = self._check_subsets()
            test_payloads.extend(subset_test_payloads)
            subset_test_payloads = self._check_handlers()
            test_payloads.extend(subset_test_payloads)
            is_valid = all([payload.is_passed for payload in test_payloads])
            setup_response = self.get_dataset_setup_response()
        except Exception as e:
            line_number = get_root_exception_line_number()
            general_error = f"Something went wrong, {repr(e)} line number: {line_number}"
            is_valid = False

        return DatasetIntegParseResult(is_valid=is_valid, payloads=test_payloads, setup=setup_response,
                                       general_error=general_error)

    def _check_subsets(self) -> List[DatasetTestResultPayload]:
        results_list: List[DatasetTestResultPayload] = []
        subset_handler_list: List[SubsetHandler] = global_dataset_binder.setup_container.subsets
        for subset_handler in subset_handler_list:
            test_result = DatasetTestResultPayload(subset_handler.name)
            try:
                subset_result_list: List[SubsetResponse] = subset_handler.function()
                for i, subset_result in enumerate(subset_result_list):
                    state = DataStateType(DataStateEnum(i).name)
                    test_result.display[state.name] = str(subset_result)
                    subset_handler.data_length[state] = subset_result.length
            except Exception as e:
                line_number = get_root_exception_line_number()
                error_string = f"{repr(e)} line number: {line_number}"
                test_result.display[TestingSectionEnum.Errors.name] = error_string
                test_result.is_passed = False
            results_list.append(test_result)
        return results_list

    def _check_handlers(self) -> List[DatasetTestResultPayload]:
        subsets = self._subsets()
        subset_response_list = random.choice(list(subsets.values()))
        result_payloads: List[DatasetTestResultPayload] = []
        idx = 0  # TODO: implement get length and randomize index
        dataset_base_handlers: List[DatasetBaseHandler] = self._get_all_dataset_base_handlers()
        for dataset_base_handler in dataset_base_handlers:
            test_result = DatasetTestResultPayload(dataset_base_handler.name)
            for i, subset_response in enumerate(subset_response_list):
                state = DataStateEnum(i).name
                try:
                    raw_result = dataset_base_handler.function(idx, subset_response)
                    test_result.display[state] = str(raw_result)
                    result_shape = get_shape(raw_result)
                    test_result.shape = result_shape

                    # setting shape in setup for all encoders
                    if dataset_base_handler is InputHandler:
                        input_handler = cast(InputHandler, dataset_base_handler)
                        input_handler.shape = result_shape
                    if dataset_base_handler is GroundTruthHandler:
                        gt_handler = cast(GroundTruthHandler, dataset_base_handler)
                        gt_handler.shape = result_shape

                except Exception as e:
                    line_number = get_root_exception_line_number()
                    test_result.display[state] = f"{repr(e)} line number: {line_number}"
                    test_result.is_passed = False

            # TODO: check types
            # TODO: check for differences between results of states and add warning for that
            result_payloads.append(test_result)

        return result_payloads

    def _get_all_dataset_base_handlers(self) -> List[DatasetBaseHandler]:
        return global_dataset_binder.setup_container.inputs + global_dataset_binder.setup_container.ground_truths + \
               global_dataset_binder.setup_container.metadata

    def get_dataset_setup_response(self) -> DatasetSetup:
        setup = global_dataset_binder.setup_container
        subsets = [DatasetSubsetInstance(name=subset.name,
                                         training_length=subset.data_length.get(DataStateType.training.value),
                                         validation_length=subset.data_length.get(DataStateType.training.value),
                                         test_length=subset.data_length.get(DataStateType.training.value))
                   for subset in setup.subsets]

        inputs = [DatasetInputInstance(name=inp.name, subset=inp.subset, shape=inp.shape, type=inp.type)
                  for inp in setup.inputs]

        ground_truths = [DatasetOutputInstance(name=gt.name, subset=gt.subset, shape=gt.shape, type=gt.type,
                                               masked_input=gt.masked_input, labels=gt.labels)
                         for gt in setup.ground_truths]

        metadata = [DatasetMetadataInstance(name=metadata.name, subset=metadata.subset, type=metadata.type)
                    for metadata in setup.metadata]

        return DatasetSetup(subsets=subsets, inputs=inputs, outputs=ground_truths, metadata=metadata)

    @lru_cache()
    def _subsets(self) -> Dict[str, List[SubsetResponse]]:
        subsets: Dict[str, List[SubsetResponse]] = {}
        for subset in global_dataset_binder.setup_container.subsets:
            # TODO: add caching of subset result
            subset_result = subset.function()
            subsets[subset.name] = subset_result
        return subsets

    def _get_inputs(self, state: DataStateEnum, idx: int) -> Dict[str, np.ndarray]:
        result_agg = {}
        subsets = self._subsets()
        for input_handler in global_dataset_binder.setup_container.inputs:
            subset = subsets[input_handler.subset_name]
            subset_state = subset[state]
            input_result = input_handler.function(idx, subset_state)
            input_name = input_handler.name
            result_agg[input_name] = input_result
        return result_agg

    def _get_gt(self, state: DataStateEnum, idx: int) -> Dict[str, np.ndarray]:
        result_agg = {}
        subsets = self._subsets()
        for gt_handler in global_dataset_binder.setup_container.ground_truths:
            subset = subsets[gt_handler.subset_name]
            subset_state = subset[state]
            gt_result = gt_handler.function(idx, subset_state)
            gt_name = gt_handler.name
            result_agg[gt_name] = gt_result
        return result_agg

    def _get_metadata(self, state: DataStateEnum, idx: int) -> Dict[str, np.ndarray]:
        result_agg = {}
        subsets = self._subsets()
        for metadata_handler in global_dataset_binder.setup_container.metadata:
            subset = subsets[metadata_handler.subset_name]
            subset_state = subset[state]
            metadata_result = metadata_handler.function(idx, subset_state)
            metadata_name = metadata_handler.name
            result_agg[metadata_name] = metadata_result
        return result_agg
