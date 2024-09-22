from io import StringIO
from typing import IO


class DualStream(StringIO):
    def __init__(self, stream1: IO, stream2: StringIO):
        super().__init__()
        self.stream1 = stream1  # Usually sys.stdout
        self.stream2 = stream2  # The StringIO stream

    def write(self, s: str) -> int:
        # Write to both streams and return the length of the written string
        self.stream1.write(s)
        self.stream2.write(s)
        return len(s)

    def flush(self) -> None:
        self.stream1.flush()
        self.stream2.flush()

    def close(self) -> None:
        # Do not close sys.stdout
        self.stream2.close()

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False