# mypy: ignore-errors
import traceback
from dataclasses import dataclass

from typing import List, Tuple, Optional

from multiprocessing import Process, Queue

from code_loader.leap_loader_parallelized_base import LeapLoaderParallelizedBase
from code_loader.leaploader import LeapLoader
from code_loader.contract.enums import DataStateEnum


@dataclass
class SampleSerializableError:
    state: DataStateEnum
    index: int
    leap_script_trace: str
    exception_as_str: str


class SamplesGeneratorParallelized(LeapLoaderParallelizedBase):
    def __init__(self, code_path: str, code_entry_name: str, n_workers: Optional[int] = 2,
                 max_samples_in_queue: int = 128) -> None:
        super().__init__(code_path, code_entry_name, n_workers, max_samples_in_queue)

    def _create_and_start_process(self) -> Process:
        process = self.multiprocessing_context.Process(
            target=SamplesGeneratorParallelized._process_func,
            args=(self.code_path, self.code_entry_name, self._inputs_waiting_to_be_process,
                  self._ready_processed_results))
        process.daemon = True
        process.start()
        return process

    def _run_and_warm_first_process(self):
        process = self._create_and_start_process()
        self.processes = [process]

        # needed in order to make sure the preprocess func runs once in nonparallel
        self._start_process_inputs([(DataStateEnum.training, 0)])
        self._get_next_ready_processed_result()

    @staticmethod
    def _process_func(code_path: str, code_entry_name: str,
                      samples_to_process: Queue, ready_samples: Queue) -> None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        leap_loader = LeapLoader(code_path, code_entry_name)
        while True:
            state, idx = samples_to_process.get(block=True)
            try:
                sample = leap_loader.get_sample(state, idx)
            except Exception as e:
                leap_script_trace = traceback.format_exc().split('File "<string>"')[-1]
                ready_samples.put(SampleSerializableError(state, idx, leap_script_trace, str(e)))
                continue

            ready_samples.put(sample)

    def generate_samples(self, sample_identities: List[Tuple[DataStateEnum, int]]):
        return self.start_process_inputs(sample_identities)

