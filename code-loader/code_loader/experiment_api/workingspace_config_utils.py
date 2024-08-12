from dataclasses import dataclass
from os import path
import os
from typing import List, Optional
import yaml


@dataclass
class LocalProjectConfig:
    codeIntegrationId: Optional[str] = None
    projectId: Optional[str] = None
    secretId: Optional[str] = None
    secretManagerId: Optional[str] = None
    entryFile: Optional[str] = None
    includePatterns: Optional[List[str]] = None

# Loading workspace configuration from leap.yaml
def load_workspace_config(workspace_dir: Optional[str] = None) -> Optional[LocalProjectConfig]:
    if workspace_dir is None:
        workspace_dir = os.getcwd()
    elif not path.isabs(workspace_dir):
        workspace_dir = path.join(os.getcwd(), workspace_dir)

    file_path = path.join(workspace_dir, "leap.yaml")
    with open(file_path) as f:
        y = yaml.safe_load(f)
        return LocalProjectConfig(**y)