from dataclasses import dataclass
from os import path
import os
from typing import Optional
import yaml


@dataclass
class LocalProjectConfig:
    codeIntegrationId: Optional[str] = None
    projectId: Optional[str] = None
    secretId: Optional[str] = None
    entryFile: Optional[str] = None

# Loading workspace configuration from leap.yaml
def load_workspace_config(workspace_dir: Optional[str] = None) -> Optional[LocalProjectConfig]:
    if workspace_dir is None:
        workspace_dir = os.getcwd()
    elif not path.isabs(workspace_dir):
        workspace_dir = path.join(os.getcwd(), workspace_dir)

    file_path = path.join(workspace_dir, "leap.yaml")
    with open(file_path) as f:
        leap_yaml = yaml.safe_load(f)
        return LocalProjectConfig(
            codeIntegrationId=leap_yaml["codeIntegrationId"] if "codeIntegrationId" in leap_yaml else None,
            projectId=leap_yaml["projectId"] if "projectId" in leap_yaml else None,
            secretId=leap_yaml["secretId"] if "secretId" in leap_yaml else None,
            entryFile=leap_yaml["entryFile"] if "entryFile" in leap_yaml else None
        )