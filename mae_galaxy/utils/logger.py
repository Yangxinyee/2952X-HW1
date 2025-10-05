import csv
from pathlib import Path
from typing import Dict, List, Optional


class CSVLogger:
    def __init__(self, file_path: str, fieldnames: List[str]) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict[str, object]) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.flush()
        finally:
            self.file.close()








