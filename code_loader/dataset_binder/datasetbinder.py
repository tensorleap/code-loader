from typing import Callable, List, Optional, Dict, Any

import numpy as np
import numpy.typing as npt
import inspect

from code_loader.contract.datasetclasses import SectionCallableInterface, InputHandler, \
    GroundTruthHandler, MetadataHandler, DatasetIntegrationSetup, DecoderHandler, PreprocessResponse, \
    PreprocessHandler, DecoderCallableInterface, CustomLossHandler, CustomCallableInterface, PredictionTypeHandler
from code_loader.contract.enums import DatasetMetadataType, LeapDataType, Metric
from code_loader.decoders.default_decoders import DefaultDecoder, default_numeric_decoder, default_graph_decoder, \
    default_image_decoder, default_horizontal_bar_decoder, default_word_decoder, \
    default_image_mask_decoder, default_text_mask_decoder
from code_loader.utils import to_numpy_return_wrapper


class DatasetBinder:

    def __init__(self) -> None:
        self.setup_container = DatasetIntegrationSetup()
        self.cache_container: Dict[str, Any] = {"word_to_index": {}}
        self._decoder_names: List[str] = list()
        self._encoder_names: List[str] = list()
        self._extend_with_default_decoders()

    def _extend_with_default_decoders(self) -> None:
        self.set_decoder(DefaultDecoder.Image.value, default_image_decoder, LeapDataType.Image)
        self.set_decoder(DefaultDecoder.Graph.value, default_graph_decoder, LeapDataType.Graph)
        self.set_decoder(DefaultDecoder.Numeric.value, default_numeric_decoder, LeapDataType.Numeric)
        self.set_decoder(DefaultDecoder.HorizontalBar.value, default_horizontal_bar_decoder, LeapDataType.HorizontalBar)
        self.set_decoder(DefaultDecoder.Text.value, default_word_decoder, LeapDataType.Text)
        self.set_decoder(DefaultDecoder.ImageMask.value, default_image_mask_decoder, LeapDataType.ImageMask)
        self.set_decoder(DefaultDecoder.TextMask.value, default_text_mask_decoder, LeapDataType.TextMask)

    def set_decoder(self, name: str,
                    decoder: DecoderCallableInterface,
                    _type: LeapDataType,
                    heatmap_decoder: Optional[Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]] = None) -> None:
        arg_names = inspect.getfullargspec(decoder)[0]
        if heatmap_decoder:
            if arg_names != inspect.getfullargspec(heatmap_decoder)[0]:
                raise Exception(f'The argument names of the heatmap decoder callback must match the decoder callback '
                                f'{str(arg_names)}')
        self.setup_container.decoders.append(DecoderHandler(name, decoder, _type, arg_names, heatmap_decoder))
        self._decoder_names.append(name)

    def set_preprocess(self, function: Callable[[], List[PreprocessResponse]]) -> None:
        self.setup_container.preprocess = PreprocessHandler(function)

    def set_input(self, function: SectionCallableInterface, input_name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.inputs.append(InputHandler(input_name, function))

        self._encoder_names.append(input_name)

    def add_custom_loss(self, name: str, function: CustomCallableInterface) -> None:
        self.setup_container.custom_loss_handlers.append(CustomLossHandler(name, function))

    def create_prediction_type(self, name: str, labels: List[str], metrics: List[Metric],
                               custom_metrics: Optional[List[CustomCallableInterface]] = None) -> None:
        self.setup_container.prediction_types.append(PredictionTypeHandler(name, labels, metrics, custom_metrics))

    def set_ground_truth(self, function: SectionCallableInterface, gt_name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.ground_truths.append(GroundTruthHandler(gt_name, function))

        self._encoder_names.append(gt_name)

    def set_metadata(self, function: SectionCallableInterface,
                     metadata_type: DatasetMetadataType, name: str) -> None:
        function = to_numpy_return_wrapper(function)
        self.setup_container.metadata.append(MetadataHandler(name, function, metadata_type))
