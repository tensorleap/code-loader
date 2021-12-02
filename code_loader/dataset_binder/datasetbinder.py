from typing import Callable, List, Optional

from code_loader.contract.datasetclasses import SectionCallableInterface, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup, DecoderHandler, PreprocessResponse, \
    PreprocessHandler
from code_loader.contract.enums import DatasetInputType, DatasetOutputType, DatasetMetadataType
from code_loader.decoders.decoder_base import DecoderBase
from code_loader.utils import to_numpy_return_wrapper


class DatasetBinder:

    def __init__(self):
        self.setup_container = DatasetIntegrationSetup()

    def set_decoder(self, decoder: DecoderBase):
        self.setup_container.decoders.append(DecoderHandler(decoder))

    def set_preprocess(self, function: Callable[[], List[PreprocessResponse]]):
        self.setup_container.preprocess = PreprocessHandler(function)

    def set_input(self, function: SectionCallableInterface, input_name: str,
                  input_type: DatasetInputType, decoder_name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.inputs.append(InputHandler(input_name, function, input_type, [], decoder_name))

    def set_ground_truth(self, function: SectionCallableInterface, gt_name: str,
                         ground_truth_type: DatasetOutputType, decoder_name: str, labels: Optional[List[str]],
                         masked_input: Optional[str]) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.ground_truths.append(
            GroundTruthHandler(gt_name, function, ground_truth_type, labels, masked_input, [], decoder_name))

    def set_metadata(self, function: SectionCallableInterface, subset: str,
                     metadata_type: DatasetMetadataType, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.metadata.append(MetadataHandler(name, function, metadata_type))
