from typing import Callable, List, Optional, Dict, Any, Union

import numpy as np  # type: ignore

from code_loader.contract.datasetclasses import SectionCallableInterface, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup, DecoderHandler, PreprocessResponse, \
    PreprocessHandler, DecoderCallableInterface
from code_loader.contract.enums import DatasetMetadataType, LeapDataType
from code_loader.decoders.default_decoders import DefaultDecoder, default_numeric_decoder, default_graph_decoder, \
    default_image_decoder, default_horizontal_bar_decoder, default_word_decoder, default_mask_decoder
from code_loader.utils import to_numpy_return_wrapper


class DatasetBinder:

    def __init__(self) -> None:
        self.setup_container = DatasetIntegrationSetup()
        self.cache_container: Dict[str, Any] = {"word_to_index": {}}
        self._extend_with_default_decoders()

    def _extend_with_default_decoders(self) -> None:
        self.set_decoder(DefaultDecoder.Image.value, default_image_decoder, LeapDataType.Image)
        self.set_decoder(DefaultDecoder.Graph.value, default_graph_decoder, LeapDataType.Graph)
        self.set_decoder(DefaultDecoder.Numeric.value, default_numeric_decoder, LeapDataType.Numeric)
        self.set_decoder(DefaultDecoder.HorizontalBar.value, default_horizontal_bar_decoder, LeapDataType.HorizontalBar)
        self.set_decoder(DefaultDecoder.Text.value, default_word_decoder, LeapDataType.Text)
        self.set_decoder(DefaultDecoder.Mask.value, default_mask_decoder, LeapDataType.Mask)

    def set_decoder(self, name: str,
                    decoder: DecoderCallableInterface,
                    type: LeapDataType,
                    heatmap_decoder: Optional[Callable[[np.array], np.array]] = None) -> None:
        self.setup_container.decoders.append(DecoderHandler(name, decoder, type, heatmap_decoder))

    def set_preprocess(self, function: Callable[[], List[PreprocessResponse]]) -> None:
        self.setup_container.preprocess = PreprocessHandler(function)

    def set_input(self, function: SectionCallableInterface, input_name: str,
                  decoder_name: Union[DefaultDecoder, str]) -> None:
        function = to_numpy_return_wrapper(function)
        if isinstance(decoder_name, DefaultDecoder):
            decoder_name = decoder_name.value
        self.setup_container.inputs.append(InputHandler(input_name, function, decoder_name))

    def set_ground_truth(self, function: SectionCallableInterface, gt_name: str,
                         decoder_name: Union[DefaultDecoder, str]) -> None:
        function = to_numpy_return_wrapper(function)
        if isinstance(decoder_name, DefaultDecoder):
            decoder_name = decoder_name.value
        self.setup_container.ground_truths.append(
            GroundTruthHandler(gt_name, function, decoder_name))

    def set_metadata(self, function: SectionCallableInterface,
                     metadata_type: DatasetMetadataType, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.metadata.append(MetadataHandler(name, function, metadata_type))
