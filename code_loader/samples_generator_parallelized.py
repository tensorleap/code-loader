# mypy: ignore-errors
import multiprocessing
import traceback
from dataclasses import dataclass
from functools import lru_cache
from queue import Empty
from threading import Thread
from typing import List, Tuple, Optional
from multiprocessing import Process, Queue
import psutil
from code_loader.leaploader import LeapLoader
from code_loader.contract.enums import DataStateEnum


@dataclass
class SampleSerializableError:
    state: DataStateEnum
    index: int
    leap_script_trace: str
    exception_as_str: str


class SamplesGeneratorParallelized:
    def __init__(self, code_path: str, code_entry_name: str,
                 n_workers: Optional[int] = 2, max_samples_in_queue: int = 128) -> None:
        self.code_entry_name = code_entry_name
        self.code_path = code_path

        if n_workers is not None and n_workers <= 0:
            raise Exception("need at least one worker")
        self.n_workers = n_workers
        self.max_samples_in_queue = max_samples_in_queue

        self._n_samples_to_process = 0
        self._samples_to_process: Optional[Queue] = None
        self._ready_samples: Optional[Queue] = None
        self.processes: Optional[List[Process]] = None
        self._generate_samples_thread: Optional[Thread] = None
        self._should_stop_thread = False

    def _calculate_n_workers_by_hardware(self) -> int:
        p = psutil.Process(self.processes[0].pid)
        memory_usage_in_bytes = p.memory_info().rss
        total_memory_in_bytes = psutil.virtual_memory().total

        n_workers = min(int(multiprocessing.cpu_count()),
                        int(total_memory_in_bytes * 0.7 / memory_usage_in_bytes))
        n_workers = max(n_workers, 1)
        return n_workers

    def _create_and_start_process(self) -> Process:
        process = Process(
            target=SamplesGeneratorParallelized._process_func,
            args=(self.code_path, self.code_entry_name, self._samples_to_process, self._ready_samples))
        process.daemon = True
        process.start()
        return process

    def _run_and_warm_first_process(self):
        process = self._create_and_start_process()
        self.processes = [process]

        # needed in order to make sure the preprocess func runs once in nonparallel
        self._generate_samples([(DataStateEnum.training, 0)])
        self._get_next_sample()

    @lru_cache()
    def start(self) -> None:
        self._samples_to_process = Queue(5000)
        self._ready_samples = Queue(self.max_samples_in_queue)

        self._run_and_warm_first_process()
        n_workers = self.n_workers
        if self.n_workers is None:
            n_workers = self._calculate_n_workers_by_hardware()

        for _ in range(n_workers - 1):
            self.processes.append(self._create_and_start_process())

    @staticmethod
    def _process_func(code_path: str, code_entry_name: str,
                      samples_to_process: Queue, ready_samples: Queue) -> None:
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

    def _generate_samples(self, sample_identities: List[Tuple[DataStateEnum, int]]):
        assert self._samples_to_process is not None
        assert self._ready_samples is not None

        for sample in sample_identities:
            if self._should_stop_thread:
                break
            self._n_samples_to_process += 1
            self._samples_to_process.put(sample)

    def _clear_queues(self):
        if self._generate_samples_thread is not None:
            self._should_stop_thread = True
            try:
                self._samples_to_process.get_nowait()
                self._n_samples_to_process -= 1
            except Empty:
                pass
            self._generate_samples_thread.join()
        while not self._samples_to_process.empty():
            try:
                self._samples_to_process.get_nowait()
                self._n_samples_to_process -= 1
            except Empty:
                pass

        for _ in range(self._n_samples_to_process):
            self._get_next_sample()

        self._should_stop_thread = False

    def _get_next_sample(self):
        sample = self._ready_samples.get()
        self._n_samples_to_process -= 1
        return sample

    def generate_samples(self, sample_identities: List[Tuple[DataStateEnum, int]]):
        self.start()

        self._clear_queues()

        self._generate_samples_thread = Thread(target=self._generate_samples, args=(sample_identities,))
        self._generate_samples_thread.start()
        return self._get_next_sample

    @staticmethod
    def _release_queue(queue: Queue):
        assert queue is not None
        queue.close()
        queue.join_thread()

    def release(self) -> None:
        if self.processes is None:
            return
        self._clear_queues()

        self._release_queue(self._samples_to_process)
        self._release_queue(self._ready_samples)

        for process in self.processes:
            process.terminate()
            process.kill()
            process.join()
            process.close()

        self.processes = None

    def __del__(self) -> None:
        self.release()
