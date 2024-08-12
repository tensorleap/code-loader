
from typing import Any, Dict, List, Optional
from code_loader.experiment_api.epoch import Epoch
from code_loader.experiment_api.api import Api, SetExperimentNotesRequest, StartExperimentRequest
from code_loader.experiment_api.experiment_context import ExperimentContext
from code_loader.experiment_api.types import Metrics
from code_loader.experiment_api.workingspace_config_utils import load_workspace_config
from code_loader.experiment_api.client import  Client

        
class Experiment:
    def __init__(self, ctx: ExperimentContext):
        self.ctx = ctx

    def init_epoch(self, epoch: int) -> Epoch:
        return Epoch(self.ctx, epoch)
    
    def log_epoch(self, epoch: int, metrics: Optional[Metrics] = None, model_path: Optional[str] = None, tags: List[str] = ['latest'])-> None:
        epoch_o = self.init_epoch(epoch)
        if metrics is not None:
            epoch_o.set_metrics(metrics)
        epoch_o.log(model_path, tags)

    def set_notes(self, notes: Dict[str, Any])-> None:
        print(f"Setting experiment({self.ctx.experiment_id}) notes")
        self.ctx.api.set_experiment_notes(SetExperimentNotesRequest(
            experimentId=self.ctx.experiment_id,
            projectId=self.ctx.project_id,
            notes=notes
        ))
    
def init_experiment(experimentName: str, description: str, working_dir: Optional[str] = None, client: Optional[Client] = None) -> 'Experiment':
    if client is None:
        client = Client()

    api = Api(client)
    
    workspace_config = load_workspace_config(working_dir)
    if workspace_config is None or workspace_config.projectId is None:
        raise Exception("No leap workspace config found or projectId is missing, make sure you are in a leap workspace directory or provide a working_dir")

    result = api.start_experiment(StartExperimentRequest(
        projectId=workspace_config.projectId,
        experimentName=experimentName,
        description=description,
        codeIntegrationVersionId=workspace_config.codeIntegrationId
    ))
    ctx = ExperimentContext(api, result.projectId, result.versionId, result.experimentId)
    return Experiment(ctx)
