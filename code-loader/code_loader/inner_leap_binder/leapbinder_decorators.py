# mypy: ignore-errors

from typing import Optional, Union, Callable, List

import numpy as np
import numpy.typing as npt

from code_loader.contract.datasetclasses import CustomCallableInterfaceMultiArgs, \
    CustomMultipleReturnCallableInterfaceMultiArgs, ConfusionMatrixCallableInterfaceMultiArgs, CustomCallableInterface, \
    VisualizerCallableInterface, MetadataSectionCallableInterface, PreprocessResponse, SectionCallableInterface, \
    ConfusionMatrixElement
from code_loader.contract.enums import MetricDirection, LeapDataType
from code_loader import leap_binder
from code_loader.contract.visualizer_classes import LeapImage, LeapImageMask, LeapTextMask, LeapText, LeapGraph, \
    LeapHorizontalBar, LeapImageWithBBox, LeapImageWithHeatmap


def tensorleap_custom_metric(name: str, direction: Optional[MetricDirection] = MetricDirection.Downward):
    def decorating_function(
            user_function: Union[CustomCallableInterfaceMultiArgs,
            CustomMultipleReturnCallableInterfaceMultiArgs,
            ConfusionMatrixCallableInterfaceMultiArgs]
    ):
        for metric_handler in leap_binder.setup_container.metrics:
            if metric_handler.name == name:
                raise Exception(f'Metric with name {name} already exists. '
                                f'Please choose another')

        leap_binder.add_custom_metric(user_function, name, direction)

        def _validate_input_args(*args, **kwargs) -> None:
            for i, arg in enumerate(args):
                assert isinstance(arg, np.ndarray), (f'tensorleap_custom_metric validation failed: '
                                                     f'Argument #{i} should be a numpy array. Got {type(arg)}.')
                if leap_binder.batch_size_to_validate:
                    assert arg.shape[0] == leap_binder.batch_size_to_validate, \
                        (f'tensorleap_custom_metric validation failed: Argument #{i} '
                         f'first dim should be as the batch size. Got {arg.shape[0]} '
                         f'instead of {leap_binder.batch_size_to_validate}')

            for _arg_name, arg in kwargs.items():
                assert isinstance(arg, np.ndarray), (f'tensorleap_custom_metric validation failed: '
                                                     f'Argument {_arg_name} should be a numpy array. Got {type(arg)}.')
                if leap_binder.batch_size_to_validate:
                    assert arg.shape[0] == leap_binder.batch_size_to_validate, \
                        (f'tensorleap_custom_metric validation failed: Argument {_arg_name} '
                         f'first dim should be as the batch size. Got {arg.shape[0]} '
                         f'instead of {leap_binder.batch_size_to_validate}')

        def _validate_result(result) -> None:
            supported_types_message = (f'tensorleap_custom_metric validation failed: '
                                       f'Metric has returned unsupported type. Supported types are List[float], '
                                       f'List[List[ConfusionMatrixElement]], NDArray[np.float32]. ')

            if isinstance(result, list):
                if isinstance(result[0], list):
                    assert isinstance(result[0][0], ConfusionMatrixElement), \
                        f'{supported_types_message}Got List[List[{type(result[0][0])}]].'
                else:
                    assert isinstance(result[0], float), f'{supported_types_message}Got List[{type(result[0])}].'

            else:
                assert isinstance(result, np.ndarray), f'{supported_types_message}Got {type(result)}.'
                assert len(result.shape) == 1, (f'tensorleap_custom_metric validation failed: '
                                                f'The return shape should be 1D. Got {len(result.shape)}D.')
            if leap_binder.batch_size_to_validate:
                assert len(result) == leap_binder.batch_size_to_validate, \
                    f'tensorleap_custom_metrix validation failed: The return len should be as the batch size.'

        def inner(*args, **kwargs):
            _validate_input_args(*args, **kwargs)
            result = user_function(*args, **kwargs)
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_custom_visualizer(name: str, visualizer_type: LeapDataType,
                                 heatmap_function: Optional[Callable[..., npt.NDArray[np.float32]]] = None):
    def decorating_function(user_function: VisualizerCallableInterface):
        for viz_handler in leap_binder.setup_container.visualizers:
            if viz_handler.name == name:
                raise Exception(f'Visualizer with name {name} already exists. '
                                f'Please choose another')

        leap_binder.set_visualizer(user_function, name, visualizer_type, heatmap_function)

        def _validate_input_args(*args, **kwargs):
            for i, arg in enumerate(args):
                assert isinstance(arg, np.ndarray), (f'tensorleap_custom_visualizer validation failed: '
                                                     f'Argument #{i} should be a numpy array. Got {type(arg)}.')
                if leap_binder.batch_size_to_validate:
                    assert arg.shape[0] != leap_binder.batch_size_to_validate, \
                        (f'tensorleap_custom_visualizer validation failed: '
                         f'Argument #{i} should be without batch dimension. ')

            for _arg_name, arg in kwargs.items():
                assert isinstance(arg, np.ndarray), (f'tensorleap_custom_visualizer validation failed: '
                                                     f'Argument {_arg_name} should be a numpy array. Got {type(arg)}.')
                if leap_binder.batch_size_to_validate:
                    assert arg.shape[0] != leap_binder.batch_size_to_validate, \
                        (f'tensorleap_custom_visualizer validation failed: Argument {_arg_name} '
                         f'should be without batch dimension. ')

        def _validate_result(result):
            result_type_map = {
                LeapDataType.Image: LeapImage,
                LeapDataType.ImageMask: LeapImageMask,
                LeapDataType.TextMask: LeapTextMask,
                LeapDataType.Text: LeapText,
                LeapDataType.Graph: LeapGraph,
                LeapDataType.HorizontalBar: LeapHorizontalBar,
                LeapDataType.ImageWithBBox: LeapImageWithBBox,
                LeapDataType.ImageWithHeatmap: LeapImageWithHeatmap
            }
            assert isinstance(result, result_type_map[visualizer_type]), \
                (f'tensorleap_custom_visualizer validation failed: '
                 f'The return type should be {result_type_map[visualizer_type]}. Got {type(result)}.')

        def inner(*args, **kwargs):
            _validate_input_args(*args, **kwargs)
            result = user_function(*args, **kwargs)
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_metadata(name: str):
    def decorating_function(user_function: MetadataSectionCallableInterface):
        for metadata_handler in leap_binder.setup_container.metadata:
            if metadata_handler.name == name:
                raise Exception(f'Metadata with name {name} already exists. '
                                f'Please choose another')

        leap_binder.set_metadata(user_function, name)

        def _validate_input_args(sample_id: Union[int, str], preprocess_response: PreprocessResponse):
            assert isinstance(sample_id, (int, str)), \
                (f'tensorleap_metadata validation failed: '
                 f'Argument sample_id should be either int or str. Got {type(sample_id)}.')
            assert isinstance(preprocess_response, PreprocessResponse), \
                (f'tensorleap_metadata validation failed: '
                 f'Argument preprocess_response should be a PreprocessResponse. Got {type(preprocess_response)}.')
            assert type(sample_id) == preprocess_response.sample_id_type, \
                (f'tensorleap_metadata validation failed: '
                 f'Argument sample_id should be as the same type as defined in the preprocess response '
                 f'{preprocess_response.sample_id_type}. Got {type(sample_id)}.')

        def _validate_result(result):
            supported_result_types = (int, str, bool, float, dict, np.floating,
                                      np.bool_, np.unsignedinteger, np.signedinteger, np.integer)
            assert isinstance(result, supported_result_types), \
                (f'tensorleap_metadata validation failed: '
                 f'Unsupported return type. Got {type(result)}. should be any of {str(supported_result_types)}')
            if isinstance(result, dict):
                for key, value in result.items():
                    assert isinstance(key, str), \
                        (f'tensorleap_metadata validation failed: '
                         f'Keys in the return dict should be of type str. Got {type(key)}.')
                    assert isinstance(value, supported_result_types), \
                        (f'tensorleap_metadata validation failed: '
                         f'Values in the return dict should be of type {str(supported_result_types)}. Got {type(value)}.')

        def inner(sample_id, preprocess_response):
            _validate_input_args(sample_id, preprocess_response)
            result = user_function(sample_id, preprocess_response)
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_preprocess():
    def decorating_function(user_function: Callable[[], List[PreprocessResponse]]):
        leap_binder.set_preprocess(user_function)

        def _validate_input_args(*args, **kwargs):
            assert len(args) == 0 and len(kwargs) == 0, \
                (f'tensorleap_preprocess validation failed: '
                 f'The function should not take any arguments. Got {args} and {kwargs}.')

        def _validate_result(result):
            assert isinstance(result, list), \
                (f'tensorleap_preprocess validation failed: '
                 f'The return type should be a list. Got {type(result)}.')
            for i, response in enumerate(result):
                assert isinstance(response, PreprocessResponse), \
                    (f'tensorleap_preprocess validation failed: '
                     f'Element #{i} in the return list should be a PreprocessResponse. Got {type(response)}.')
            assert len(set(result)) == len(result), \
                (f'tensorleap_preprocess validation failed: '
                 f'The return list should not contain duplicate PreprocessResponse objects.')

        def inner(*args, **kwargs):
            _validate_input_args(*args, **kwargs)
            result = user_function()
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_unlabeled_preprocess():
    def decorating_function(user_function: Callable[[], PreprocessResponse]):
        leap_binder.set_unlabeled_data_preprocess(user_function)

        def _validate_input_args(*args, **kwargs):
            assert len(args) == 0 and len(kwargs) == 0, \
                (f'tensorleap_unlabeled_preprocess validation failed: '
                 f'The function should not take any arguments. Got {args} and {kwargs}.')

        def _validate_result(result):
            assert isinstance(result, PreprocessResponse), \
                (f'tensorleap_unlabeled_preprocess validation failed: '
                 f'The return type should be a PreprocessResponse. Got {type(result)}.')

        def inner(*args, **kwargs):
            _validate_input_args(*args, **kwargs)
            result = user_function()
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_input_encoder(name: str):
    def decorating_function(user_function: SectionCallableInterface):
        for input_handler in leap_binder.setup_container.inputs:
            if input_handler.name == name:
                raise Exception(f'Input with name {name} already exists. '
                                f'Please choose another')

        leap_binder.set_input(user_function, name)

        def _validate_input_args(sample_id: Union[int, str], preprocess_response: PreprocessResponse):
            assert isinstance(sample_id, (int, str)), \
                (f'tensorleap_input_encoder validation failed: '
                 f'Argument sample_id should be either int or str. Got {type(sample_id)}.')
            assert isinstance(preprocess_response, PreprocessResponse), \
                (f'tensorleap_input_encoder validation failed: '
                 f'Argument preprocess_response should be a PreprocessResponse. Got {type(preprocess_response)}.')
            assert type(sample_id) == preprocess_response.sample_id_type, \
                (f'tensorleap_input_encoder validation failed: '
                 f'Argument sample_id should be as the same type as defined in the preprocess response '
                 f'{preprocess_response.sample_id_type}. Got {type(sample_id)}.')

        def _validate_result(result):
            assert isinstance(result, np.ndarray), \
                (f'tensorleap_input_encoder validation failed: '
                 f'Unsupported return type. Should be a numpy array. Got {type(result)}.')
            assert result.dtype == np.float32, \
                (f'tensorleap_input_encoder validation failed: '
                 f'The return type should be a numpy array of type float32. Got {result.dtype}.')

        def inner(sample_id, preprocess_response):
            _validate_input_args(sample_id, preprocess_response)
            result = user_function(sample_id, preprocess_response)
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_gt_encoder(name: str):
    def decorating_function(user_function: SectionCallableInterface):
        for gt_handler in leap_binder.setup_container.ground_truths:
            if gt_handler.name == name:
                raise Exception(f'GT with name {name} already exists. '
                                f'Please choose another')

        leap_binder.set_ground_truth(user_function, name)

        def _validate_input_args(sample_id: Union[int, str], preprocess_response: PreprocessResponse):
            assert isinstance(sample_id, (int, str)), \
                (f'tensorleap_gt_encoder validation failed: '
                 f'Argument sample_id should be either int or str. Got {type(sample_id)}.')
            assert isinstance(preprocess_response, PreprocessResponse), \
                (f'tensorleap_gt_encoder validation failed: '
                 f'Argument preprocess_response should be a PreprocessResponse. Got {type(preprocess_response)}.')
            assert type(sample_id) == preprocess_response.sample_id_type, \
                (f'tensorleap_gt_encoder validation failed: '
                 f'Argument sample_id should be as the same type as defined in the preprocess response '
                 f'{preprocess_response.sample_id_type}. Got {type(sample_id)}.')

        def _validate_result(result):
            assert isinstance(result, np.ndarray), \
                (f'tensorleap_gt_encoder validation failed: '
                 f'Unsupported return type. Should be a numpy array. Got {type(result)}.')
            assert result.dtype == np.float32, \
                (f'tensorleap_gt_encoder validation failed: '
                 f'The return type should be a numpy array of type float32. Got {result.dtype}.')

        def inner(sample_id, preprocess_response):
            _validate_input_args(sample_id, preprocess_response)
            result = user_function(sample_id, preprocess_response)
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_custom_loss(name: str):
    def decorating_function(user_function: CustomCallableInterface):
        for loss_handler in leap_binder.setup_container.custom_loss_handlers:
            if loss_handler.name == name:
                raise Exception(f'Custom loss with name {name} already exists. '
                                f'Please choose another')

        leap_binder.add_custom_loss(user_function, name)

        def _validate_input_args(*args, **kwargs):
            try:
                import tensorflow as tf
            except ImportError as e:
                raise Exception('the input arguments of the custom loss function should be tensorflow tensors') from e

            for i, arg in enumerate(args):
                assert isinstance(arg, tf.Tensor), (f'tensorleap_custom_loss validation failed: '
                                                    f'Argument #{i} should be a tensorflow tensor. Got {type(arg)}.')
            for _arg_name, arg in kwargs.items():
                assert isinstance(arg, tf.Tensor), (f'tensorleap_custom_loss validation failed: '
                                                    f'Argument {_arg_name} should be a tensorflow tensor. Got {type(arg)}.')

        def _validate_result(result):
            try:
                import tensorflow as tf
            except ImportError:
                raise Exception('the input arguments of the custom loss function should be tensorflow tensors')

            assert isinstance(result, (np.ndarray, tf.Tensor)), \
                (f'tensorleap_custom_loss validation failed: '
                 f'The return type should be a numpy array or a tensorflow tensor. Got {type(result)}.')

        def inner(sample_id, preprocess_response):
            _validate_input_args(sample_id, preprocess_response)
            result = user_function(sample_id, preprocess_response)
            _validate_result(result)
            return result

        return inner

    return decorating_function


def tensorleap_custom_layer(name: str):
    def decorating_function(custom_layer):
        for custom_layer_handler in leap_binder.setup_container.custom_layers.values():
            if custom_layer_handler.name == name:
                raise Exception(f'Custom Layer with name {name} already exists. '
                                f'Please choose another')

        try:
            import tensorflow as tf
        except ImportError as e:
            raise Exception('Custom layer should be inherit from tf.keras.layers.Layer') from e

        if not issubclass(custom_layer, tf.keras.layers.Layer):
            raise Exception('Custom layer should be inherit from tf.keras.layers.Layer')

        leap_binder.set_custom_layer(custom_layer, name)

        return custom_layer

    return decorating_function
