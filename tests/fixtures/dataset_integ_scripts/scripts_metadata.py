from typing import Dict

import pytest


@pytest.fixture
def word_idx_dataset_params() -> Dict[str, str]:
    from .simple_no_cloud_dataset_wt_word_idx import input_name, word_to_index_value
    return {
        "input_name": input_name,
        "word_to_index_value": word_to_index_value,
    }
