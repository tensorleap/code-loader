from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Union
from urllib.parse import urljoin
from code_loader.experiment_api.types import ApiMetricValue, MetricValue, NumericMetricValue, StringMetricValue
import requests


def upload_file(url: str, file_path: str)-> None:
    with open(file_path, "rb") as f:
        requests.put(url, data=f, timeout=12_000)

def to_dict_no_none(data: Any)-> Union[Dict[str, Any], Any]:
    if is_dataclass(data):
        data = asdict(data)
    if isinstance(data, dict):
        return {k: to_dict_no_none(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [to_dict_no_none(item) for item in data]
    else:
        return data

def join_url(base_url: str, post_path: str)-> str:
    if not base_url.endswith('/'):
        base_url += '/'
    if post_path.startswith('/'):
        post_path = post_path[1:]
    return urljoin(base_url, post_path)

def to_api_metric_value(value: MetricValue) -> ApiMetricValue:
    if isinstance(value, float) or isinstance(value, int):
        return NumericMetricValue(value=value)
    elif isinstance(value, str):
        return StringMetricValue(value=value)
    else:
        raise Exception(f"Unsupported metric value type: {type(value)}")