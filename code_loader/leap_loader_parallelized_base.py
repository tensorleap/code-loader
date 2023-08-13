# mypy: ignore-errors
import multiprocessing
from abc import ABC, abstractmethod
from functools import lru_cache
from queue import Empty
from threading import Thread
from typing import List, Optional, Any
from multiprocessing import Process, Queue
import psutil


class LeapLoaderParallelizedBase(ABC):
    def __init__(self, code_path: str, code_entry_name: str,
                 n_workers: Optional[int] = 2, max_ready_results_in_queue: int = 128,
                 multiprocessing_context: Optional[str] = None) -> None:
        self.multiprocessing_context = multiprocessing
        if multiprocessing_context is not None:
            self.multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

        self.code_entry_name = code_entry_name
        self.code_path = code_path

        if n_workers is not None and n_workers <= 0:
            raise Exception("need at least one worker")
        self.n_workers = n_workers
        self.max_ready_results_in_queue = max_ready_results_in_queue

        self._n_inputs_waiting_to_be_process = 0
        self._inputs_waiting_to_be_process: Optional[Queue] = None
        self._ready_processed_results: Optional[Queue] = None
        self.processes: Optional[List[Process]] = None
        self._generate_inputs_thread: Optional[Thread] = None
        self._should_stop_thread = False

    def _calculate_n_workers_by_hardware(self) -> int:
        p = psutil.Process(self.processes[0].pid)
        memory_usage_in_bytes = p.memory_info().rss
        total_memory_in_bytes = psutil.virtual_memory().total

        n_workers = min(int(multiprocessing.cpu_count()),
                        int(total_memory_in_bytes * 0.5 / memory_usage_in_bytes))
        n_workers = max(n_workers, 1)
        return n_workers

    @abstractmethod
    def _create_and_start_process(self) -> Process:
        pass

    def _run_and_warm_first_process(self):
        pass

    @lru_cache()
    def start(self) -> None:
        self._inputs_waiting_to_be_process = self.multiprocessing_context.Queue(5000)
        self._ready_processed_results = self.multiprocessing_context.Queue(self.max_ready_results_in_queue)

        self._run_and_warm_first_process()
        n_workers = self.n_workers
        if self.n_workers is None:
            n_workers = self._calculate_n_workers_by_hardware()

        if self.processes is None:
            self.processes = []
        for _ in range(n_workers):
            self.processes.append(self._create_and_start_process())

    def _start_process_inputs(self, inputs: List[Any]):
        assert self._inputs_waiting_to_be_process is not None
        assert self._ready_processed_results is not None

        for _input in inputs:
            if self._should_stop_thread:
                break
            self._n_inputs_waiting_to_be_process += 1
            self._inputs_waiting_to_be_process.put(_input)

    def _clear_queues(self):
        if self._generate_inputs_thread is not None:
            self._should_stop_thread = True
            try:
                self._inputs_waiting_to_be_process.get_nowait()
                self._n_inputs_waiting_to_be_process -= 1
            except Empty:
                pass
            self._generate_inputs_thread.join()
        while not self._inputs_waiting_to_be_process.empty():
            try:
                self._inputs_waiting_to_be_process.get_nowait()
                self._n_inputs_waiting_to_be_process -= 1
            except Empty:
                pass

        for _ in range(self._n_inputs_waiting_to_be_process):
            self._get_next_ready_processed_result()

        self._should_stop_thread = False

    def _get_next_ready_processed_result(self):
        result = self._ready_processed_results.get()
        self._n_inputs_waiting_to_be_process -= 1
        return result

    def start_process_inputs(self, inputs: List[Any]):
        self.start()

        self._clear_queues()

        self._generate_inputs_thread = Thread(target=self._start_process_inputs, args=(inputs,))
        self._generate_inputs_thread.start()
        return self._get_next_ready_processed_result

    @staticmethod
    def _release_queue(queue: Queue):
        assert queue is not None
        queue.close()
        queue.join_thread()

    def release(self) -> None:
        if self.processes is None:
            return
        self._clear_queues()

        self._release_queue(self._inputs_waiting_to_be_process)
        self._release_queue(self._ready_processed_results)

        for process in self.processes:
            process.terminate()
            process.kill()
            process.join()
            process.close()

        self.processes = None

    def __del__(self) -> None:
        self.release()
