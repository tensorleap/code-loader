import numpy as np  # type: ignore

from code_loader.contract.datasetclasses import SectionCallableInterface, SubsetResponse


def to_numpy_return_wrapper(encoder_function: SectionCallableInterface) -> SectionCallableInterface:
    def numpy_encoder_function(idx: int, samples: SubsetResponse) -> np.ndarray:
        result = encoder_function(idx, samples)
        numpy_result = np.array(result)
        return numpy_result

    return numpy_encoder_function
