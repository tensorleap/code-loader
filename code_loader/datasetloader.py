from typing import Dict


class DatasetLoader:

    def __init__(self, dataset_script: str):
        self.dataset_script = dataset_script
        self.index_dict: Dict[str, int] = {}
        self.global_variables = {'index_dict': self.index_dict}

    def exec_script(self):
        exec(self.dataset_script, self.global_variables)
