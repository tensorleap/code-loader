
from typing import List, Optional
from code_loader.experiment_api.experiment_context import ExperimentContext
from code_loader.experiment_api.types import Metrics
from code_loader.experiment_api.utils import to_api_metric_value, upload_file
from code_loader.experiment_api.api import AddExternalEpochDataRequest, GetUploadModelSignedUrlRequest, TagModelRequest


class Epoch:
    def __init__(self, ctx: ExperimentContext, epoch: int):
        self.experiment = ExperimentContext
        self.epoch = epoch
        self.metrics: Metrics = {}
        self.ctx = ctx

    def add_metric(self, name: str, value: float)-> None:
        self.metrics[name] = value

    def set_metrics(self, metrics: Metrics)-> None:
        self.metrics = metrics

    def _upload_model(self, modelFilePath: str)-> None:
        allowed_extensions = ["h5", "onnx"]
        modelExtension = modelFilePath.split(".")[-1]
        if modelExtension not in allowed_extensions:
            raise Exception(f"Model file extension not allowed. Allowed extensions are {allowed_extensions}")
        url = self.ctx.api.get_uploaded_model_signed_url(GetUploadModelSignedUrlRequest(
            epoch=self.epoch,
            experimentId=self.ctx.experiment_id,
            versionId=self.ctx.version_id,
            projectId=self.ctx.project_id,
            fileType=modelExtension
        ))
        print(f"Uploading epoch({self.epoch}) model file")
        upload_file(url.url, modelFilePath)
        print("Model file uploaded")
    
    def _tag_model(self, tags: List[str])-> None:
        print(f"Tagging epoch({self.epoch}) model")
        self.ctx.api.tag_model(TagModelRequest(
            experimentId=self.ctx.experiment_id,
            projectId=self.ctx.project_id,
            epoch=self.epoch,
            tags=tags
        ))

    def log(self, modelFilePath: Optional[str] = None, tags: List[str] = ['latest'])-> None:
        if modelFilePath is not None:
            self._upload_model(modelFilePath)

        print(f"Add metrics for epoch({self.epoch}) model")
        api_metrics ={
            key: to_api_metric_value(value) for key, value in self.metrics.items()
        }
        self.ctx.api.add_external_epoch_data(AddExternalEpochDataRequest(
            experimentId=self.ctx.experiment_id,
            projectId=self.ctx.project_id,
            epoch=self.epoch,
            metrics=api_metrics
        ))
        if modelFilePath is not None and len(tags) > 0:
            self._tag_model(tags)
   