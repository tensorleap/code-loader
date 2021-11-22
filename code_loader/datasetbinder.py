from typing import Callable, List, Optional

from code_loader.contract.datasetclasses import SubsetResponse, SectionCallableInterface, SubsetHandler, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType


class DatasetBinder:

    def __init__(self):
        self.setup_container = DatasetIntegrationSetup()

    def set_subset(self, ratio: float, function: Callable[[None], List[SubsetResponse]], name: str) -> None:
        self.setup_container.subsets.append(SubsetHandler(ratio, function, name))

    def set_input(self, function: SectionCallableInterface, subset: str,
                  input_type: DatasetInputType, name: str) -> None:
        self.setup_container.inputs.append(InputHandler(name, function, subset, input_type, []))

    def set_ground_truth(self, function: SectionCallableInterface, subset: str,
                         ground_truth_type: DatasetOutputType, name: str, labels: List[str],
                         masked_input: Optional[str]) -> None:
        self.setup_container.ground_truths.append(GroundTruthHandler(name, function, subset,
                                                                     ground_truth_type, labels, masked_input, []))

    def set_metadata(self, function: SectionCallableInterface, subset: str,
                     metadata_type: DatasetMetadataType, name: str) -> None:
        self.setup_container.metadata.append(MetadataHandler(name, function, subset, metadata_type))
