from functools import lru_cache
from typing import Dict, List, Iterable, Any

import numpy as np
import numpy.typing as npt

from code_loader.contract.datasetclasses import DatasetSample, DatasetBaseHandler, InputHandler, \
    GroundTruthHandler, PreprocessResponse, DecoderHandler, DecoderCallableReturnType
from code_loader.contract.enums import DataStateEnum, TestingSectionEnum, DataStateType
from code_loader.contract.responsedataclasses import DatasetIntegParseResult, DatasetTestResultPayload, \
    DatasetPreprocess, DatasetSetup, DatasetInputInstance, DatasetOutputInstance, DatasetMetadataInstance, \
    DecoderInstance
from code_loader.dataset_binder import global_dataset_binder
from code_loader.utils import get_root_exception_line_number, get_shape


# TODO: add handling of large return messages (usually return of large subsets)
# MAX_PUBSUB_MSG_SIZE = 500000


class DatasetLoader:
    def __init__(self, dataset_script: str):
        self.dataset_script: str = dataset_script

    @lru_cache()
    def exec_script(self) -> None:
        global_variables: Dict[Any, Any] = {}
        exec(self.dataset_script, global_variables)

    @lru_cache()
    def decoder_by_name(self) -> Dict[str, DecoderHandler]:
        self.exec_script()
        setup = global_dataset_binder.setup_container
        return {
            decoder_handler.name: decoder_handler
            for decoder_handler in setup.decoders
        }

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
            preprocess_test_payload = self._check_preprocess()
            test_payloads.append(preprocess_test_payload)
            handlers_test_payloads = self._check_handlers()
            test_payloads.extend(handlers_test_payloads)
            is_valid = all([payload.is_passed for payload in test_payloads])
            setup_response = self.get_dataset_setup_response()
        except Exception as e:
            line_number = get_root_exception_line_number()
            general_error = f"Something went wrong, {repr(e)} line number: {line_number}"
            is_valid = False

        return DatasetIntegParseResult(is_valid=is_valid, payloads=test_payloads, setup=setup_response,
                                       general_error=general_error)

    @staticmethod
    def _check_preprocess() -> DatasetTestResultPayload:
        preprocess_handler = global_dataset_binder.setup_container.preprocess
        test_result = DatasetTestResultPayload('preprocess')
        try:
            if preprocess_handler is None:
                raise Exception('None preprocess_handler')
            preprocess_result_list = preprocess_handler.function()
            for state, preprocess_result in zip(list(DataStateType), preprocess_result_list):
                state_name = state.name
                test_result.display[state_name] = str(preprocess_result.data)
                preprocess_handler.data_length[state] = preprocess_result.length
        except Exception as e:
            line_number = get_root_exception_line_number()
            error_string = f"{repr(e)} line number: {line_number}"
            test_result.display[TestingSectionEnum.Errors.name] = error_string
            test_result.is_passed = False
        return test_result

    def _check_handlers(self) -> List[DatasetTestResultPayload]:
        preprocess_result = self._preprocess_result()
        result_payloads: List[DatasetTestResultPayload] = []
        idx = 0
        dataset_base_handlers: List[DatasetBaseHandler] = self._get_all_dataset_base_handlers()
        for dataset_base_handler in dataset_base_handlers:
            test_result = DatasetTestResultPayload(dataset_base_handler.name)
            for state, preprocess_response in zip(list(DataStateEnum), preprocess_result):
                state_name = state.name
                try:
                    raw_result = dataset_base_handler.function(idx, preprocess_response)
                    test_result.display[state_name] = str(raw_result)
                    result_shape = get_shape(raw_result)
                    test_result.shape = result_shape

                    # setting shape in setup for all encoders
                    if isinstance(dataset_base_handler, (InputHandler, GroundTruthHandler)):
                        dataset_base_handler.shape = result_shape

                except Exception as e:
                    line_number = get_root_exception_line_number()
                    test_result.display[state_name] = f"{repr(e)} line number: {line_number}"
                    test_result.is_passed = False

            # TODO: check types
            # TODO: check for differences between results of states and add warning for that
            result_payloads.append(test_result)

        return result_payloads

    @staticmethod
    def _get_all_dataset_base_handlers() -> List[DatasetBaseHandler]:
        all_dataset_base_handlers: List[DatasetBaseHandler] = []
        all_dataset_base_handlers.extend(global_dataset_binder.setup_container.inputs)
        all_dataset_base_handlers.extend(global_dataset_binder.setup_container.ground_truths)
        all_dataset_base_handlers.extend(global_dataset_binder.setup_container.metadata)
        return all_dataset_base_handlers

    def run_decoder(self, decoder_name: str, input_tensors: List[npt.NDArray[np.float32]],
                    ) -> DecoderCallableReturnType:
        return self.decoder_by_name()[decoder_name].function(*input_tensors)

    def run_heatmap_decoder(self, decoder_name: str, input_heatmaps: List[npt.NDArray[np.float32]]) -> npt.NDArray[np.float32]:
        heatmap_function = self.decoder_by_name()[decoder_name].heatmap_function
        if heatmap_function is None:
            assert len(input_heatmaps) == 1
            return input_heatmaps[0]
        return heatmap_function(*input_heatmaps)

    @staticmethod
    def get_dataset_setup_response() -> DatasetSetup:
        setup = global_dataset_binder.setup_container
        assert setup.preprocess is not None
        dataset_preprocess = DatasetPreprocess(
            training_length=setup.preprocess.data_length[DataStateType.training],
            validation_length=setup.preprocess.data_length[DataStateType.validation],
            test_length=setup.preprocess.data_length.get(DataStateType.test))

        inputs = []
        for inp in setup.inputs:
            if inp.shape is None:
                raise Exception(f"cant calculate shape for input, input name:{inp.name}")
            inputs.append(DatasetInputInstance(name=inp.name, shape=inp.shape))

        ground_truths = []
        for gt in setup.ground_truths:
            if gt.shape is None:
                raise Exception(f"cant calculate shape for ground truth, gt name:{gt.name}")
            ground_truths.append(
                DatasetOutputInstance(name=gt.name, shape=gt.shape))

        metadata = [DatasetMetadataInstance(name=metadata.name, type=metadata.type)
                    for metadata in setup.metadata]

        decoders = [DecoderInstance(name=decoder_handler.name, type=decoder_handler.type)
                    for decoder_handler in setup.decoders]

        return DatasetSetup(preprocess=dataset_preprocess, inputs=inputs, outputs=ground_truths, metadata=metadata,
                            decoders=decoders, prediction_types=setup.prediction_types)

    @lru_cache()
    def _preprocess_result(self) -> List[PreprocessResponse]:
        preprocess = global_dataset_binder.setup_container.preprocess
        # TODO: add caching of subset result
        assert preprocess is not None
        preprocess_result = preprocess.function()

        return preprocess_result

    def _get_dataset_handlers(
            self, handlers: Iterable[DatasetBaseHandler], state: DataStateEnum, idx: int) -> Dict[str, npt.NDArray[np.float32]]:
        result_agg = {}
        preprocess_result = self._preprocess_result()
        preprocess_state = preprocess_result[state]
        for handler in handlers:
            handler_result = handler.function(idx, preprocess_state)
            handler_name = handler.name
            result_agg[handler_name] = handler_result
        return result_agg

    def _get_inputs(self, state: DataStateEnum, idx: int) -> Dict[str, npt.NDArray[np.float32]]:
        return self._get_dataset_handlers(global_dataset_binder.setup_container.inputs, state, idx)

    def _get_gt(self, state: DataStateEnum, idx: int) -> Dict[str, npt.NDArray[np.float32]]:
        return self._get_dataset_handlers(global_dataset_binder.setup_container.ground_truths, state, idx)

    def _get_metadata(self, state: DataStateEnum, idx: int) -> Dict[str, npt.NDArray[np.float32]]:
        return self._get_dataset_handlers(global_dataset_binder.setup_container.metadata, state, idx)
