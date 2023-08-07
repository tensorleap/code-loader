# mypy: ignore-errors
from typing import Optional, List, Tuple, Dict
from multiprocessing import Process, Queue

import numpy as np

from code_loader.leap_loader_parallelized_base import LeapLoaderParallelizedBase
from dataclasses import dataclass
from code_loader.leaploader import LeapLoader


@dataclass
class VisualizerSerializableError:
    visualizer_id: str
    visualizer_name: str
    index_in_batch: int
    exception_as_str: str


class VisualizerCalculatorParallelized(LeapLoaderParallelizedBase):
    def __init__(self, code_path: str, code_entry_name: str, n_workers: Optional[int] = 2,
                 max_samples_in_queue: int = 128) -> None:
        super().__init__(code_path, code_entry_name, n_workers, max_samples_in_queue, "spawn")

    @staticmethod
    def _process_func(code_path: str, code_entry_name: str,
                      visualizers_to_process: Queue, ready_visualizations: Queue) -> None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        leap_loader = LeapLoader(code_path, code_entry_name)

        # running preprocessing to sync preprocessing in main thread (can be valuable when preprocess is filling a
        # global param that visualizer is using)
        leap_loader._preprocess_result()
        leap_loader._preprocess_result.cache_clear()

        while True:
            index_in_batch, visualizer_id, visualizer_name, input_arg_name_to_tensor = \
                visualizers_to_process.get(block=True)
            try:
                visualizer_result = \
                    leap_loader.visualizer_by_name()[visualizer_name].function(**input_arg_name_to_tensor)
            except Exception as e:
                ready_visualizations.put(VisualizerSerializableError(
                    visualizer_id, visualizer_name, index_in_batch, str(e)))
                continue

            ready_visualizations.put((index_in_batch, visualizer_id, visualizer_result))

    def _create_and_start_process(self) -> Process:
        process = self.multiprocessing_context.Process(
            target=VisualizerCalculatorParallelized._process_func,
            args=(self.code_path, self.code_entry_name, self._inputs_waiting_to_be_process,
                  self._ready_processed_results))
        process.daemon = True
        process.start()
        return process

    def calculate_visualizers(self, input_arg_name_to_tensor_list: List[Tuple[int, str, str, Dict[str, np.array]]]):
        return self.start_process_inputs(input_arg_name_to_tensor_list)
