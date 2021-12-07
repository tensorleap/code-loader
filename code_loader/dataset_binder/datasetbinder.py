from typing import Callable, List, Optional, Dict, Any

from code_loader.contract.datasetclasses import SubsetResponse, SectionCallableInterface, SubsetHandler, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType
from code_loader.utils import to_numpy_return_wrapper


class DatasetBinder:

    def __init__(self) -> None:
        self.setup_container = DatasetIntegrationSetup()
        self.cache_container: Dict[str, Any] = {"word_to_index": {}}

    def set_subset(self, function: Callable[[], List[SubsetResponse]], name: str) -> None:
        self.setup_container.subsets.append(SubsetHandler(function, name))

    def set_input(self, function: SectionCallableInterface, subset: str,
                  input_type: DatasetInputType, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.inputs.append(InputHandler(name, function, subset, input_type, []))

    def set_ground_truth(self, function: SectionCallableInterface, subset: str,
                         ground_truth_type: DatasetOutputType, name: str, labels: Optional[List[str]],
                         masked_input: Optional[str]) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.ground_truths.append(GroundTruthHandler(name, function, subset,
                                                                     ground_truth_type, labels, masked_input, []))

    def set_metadata(self, function: SectionCallableInterface, subset: str,
                     metadata_type: DatasetMetadataType, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.metadata.append(MetadataHandler(name, function, subset, metadata_type))
