from functools import lru_cache
from typing import Dict, List

import numpy as np  # type: ignore

from code_loader.dataset_binder import global_dataset_binder
from code_loader.contract.datasetclasses import SubsetResponse, DatasetSample, DatasetIntegrationSetup
from code_loader.contract.enums import DataStateEnum


class DatasetLoader:

    def __init__(self, dataset_script: str):
        self.dataset_script: str = dataset_script
        self.index_dict: Dict[str, int] = {}
        self.global_variables = {'index_dict': self.index_dict}
        self.executed_script: bool = False

    def exec_script(self):
        if not self.executed_script:
            exec(self.dataset_script, self.global_variables)
            self.executed_script = True

    def get_sample(self, state: DataStateEnum, idx: int) -> DatasetSample:
        self.exec_script()
        sample = DatasetSample(inputs=self._get_inputs(state, idx),
                               gt=self._get_gt(state, idx),
                               metadata=self._get_metadata(state, idx),
                               index=idx,
                               state=state)
        return sample

    @lru_cache()
    def _subsets(self) -> Dict[str, List[SubsetResponse]]:
        subsets: Dict[str, List[SubsetResponse]] = {}
        for subset in global_dataset_binder.setup_container.subsets:
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
