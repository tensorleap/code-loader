from typing import Optional, Union

import numpy as np

from code_loader.contract.datasetclasses import CustomCallableInterfaceMultiArgs, \
    CustomMultipleReturnCallableInterfaceMultiArgs, ConfusionMatrixCallableInterfaceMultiArgs
from code_loader.contract.enums import MetricDirection
from code_loader import leap_binder


def tensorleap_custom_metric(name: str, direction: Optional[MetricDirection] = MetricDirection.Downward):
    def decorating_function(
            user_function: Union[CustomCallableInterfaceMultiArgs,
                                 CustomMultipleReturnCallableInterfaceMultiArgs,
                                 ConfusionMatrixCallableInterfaceMultiArgs]
    ):

        leap_binder.add_custom_metric(user_function, name, direction)

        def _validate_custom_metric_input_args(*args, **kwargs):
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

        def _validate_custom_metric_result(result):
            assert isinstance(result, np.ndarray), (f'tensorleap_custom_metric validation failed: '
                                                    f'The return type should be a numpy array. Got {type(result)}.')
            assert len(result.shape) == 1, (f'tensorleap_custom_metric validation failed: '
                                            f'The return shape should be 1D. Got {len(result.shape)}D.')
            if leap_binder.batch_size_to_validate:
                assert result.shape[0] == leap_binder.batch_size_to_validate, \
                    f'tensorleap_custom_metric validation failed: The return len should be as the batch size.'

        def inner(*args, **kwargs):
            _validate_custom_metric_input_args(*args, **kwargs)
            result = user_function(*args, **kwargs)
            _validate_custom_metric_result(result)
            return result

        return inner

    return decorating_function
