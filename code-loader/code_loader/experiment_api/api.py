
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from code_loader.experiment_api.client import Client
from code_loader.experiment_api.types import ApiMetrics


@dataclass
class StartExperimentRequest:
    projectId: str
    experimentName: str
    description: str
    removeUntaggedUploadedModels: bool = True
    codeIntegrationVersionId: Optional[str] = None

@dataclass
class StartExperimentResponse:
    projectId: str
    versionId: str
    experimentId: str

@dataclass
class GetUploadModelSignedUrlRequest:
    epoch: int
    experimentId: str
    versionId: str
    projectId: str
    fileType: str
    origin: Optional[str] = None

@dataclass
class GetUploadModelSignedUrlResponse:
    url: str
    fileName: str

@dataclass
class AddExternalEpochDataRequest:
  projectId: str
  experimentId: str
  epoch: int
  metrics: ApiMetrics
  force: bool = False

@dataclass
class TagModelRequest:
  projectId: str
  experimentId: str
  epoch: int
  tags: List[str]

@dataclass
class SetExperimentNotesRequest:
    projectId: str
    experimentId: str
    notes: Dict[str, Any]

class Api:
    def __init__(self, client: Client):
        self.client = client
    
    def start_experiment(self, data: StartExperimentRequest) -> StartExperimentResponse:
        response = self.client.post('/versions/startExperiment', data)
        self.client.check_response(response)
        return StartExperimentResponse(**response.json())
    
    def get_uploaded_model_signed_url(self, data: GetUploadModelSignedUrlRequest)-> GetUploadModelSignedUrlResponse:
        response = self.client.post('/versions/getUploadModelSignedUrl', data)
        self.client.check_response(response)
        return GetUploadModelSignedUrlResponse(**response.json())
    
    def add_external_epoch_data(self, data: AddExternalEpochDataRequest)-> None:
        response = self.client.post('/externalepochdata/addExternalEpochData', data)
        self.client.check_response(response)
    
    def tag_model(self, data: TagModelRequest)-> None:
        response = self.client.post('/versions/tagModel', data)
        self.client.check_response(response)

    def set_experiment_notes(self, data: SetExperimentNotesRequest)-> None:
        response = self.client.post('/versions/setExperimentNotes', data)
        self.client.check_response(response)