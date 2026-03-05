"""Preprocessing raw data from Brett et al., 2024.

The raw data is available at
https://figshare.com/articles/dataset/Inner_Core_Travel_Time_Data_Reference_Model_PREM/26535394/1?file=48335947
Figshare doesn't seem to allow non-browser downloads, so the data must be downloaded manually and saved as ../data/brett2024_ic_travetimes_raw.txt
This script processes that raw data into a more convenient parquet format saved at ../data/brett2024_ic_travetimes.parquet
"""

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_PATH = DATA_DIR / "brett2024_ic_traveltimes_raw.txt"
PROCESSED_DATA_PATH = DATA_DIR / "brett2024_ic_traveltimes.parquet"

EXPECTED_SHA256 = "5e93a4abe04d7b533343eef0b859c653f310034f5258a1e3fb314455a91afed9"


def sha256(path: Path):
    """Compute the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):  # Read in 8KB chunks
            h.update(chunk)
    return h.hexdigest()


def validate_file(path: Path, expected_hash: str):
    """Validate the SHA256 hash of a file against an expected hash."""
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    logger.info(f"Validating file {path}")
    file_hash = sha256(path)
    if file_hash != expected_hash:
        raise ValueError(
            f"File {path} has hash {file_hash}, expected {expected_hash}.  Please re-download the file."
        )
    logger.info("File validation successful")


@dataclass
class DataEntry:
    """Representation of a single parsed data entry."""

    filename: str
    reference_phase: str
    delta_t: float
    inner_core_travel_time: float
    zeta: float
    in_location: list[float]
    turning_point: list[float]
    out_location: list[float]


def parse_line(line: str) -> DataEntry:
    """Parse a single line of the raw data file into a DataEntry."""
    # Remove special characters like brackets and commas
    line = line.replace("[", "").replace("]", "").replace(",", "")
    parts = line.split()
    # find the index of the reference phase
    ref_phase_index = next(
        i for i, part in enumerate(parts) if part in ["ab", "bc", "cd", "df"]
    )
    # join all parts before the reference phase as filename
    filename = ".".join(parts[:ref_phase_index])
    cleaned_parts = [filename] + parts[ref_phase_index:]
    return DataEntry(
        filename=cleaned_parts[0],
        reference_phase=cleaned_parts[1],
        delta_t=float(cleaned_parts[2]),
        inner_core_travel_time=float(cleaned_parts[3]),
        zeta=float(cleaned_parts[4]),
        in_location=[
            float(cleaned_parts[5]),
            float(cleaned_parts[6]),
            float(cleaned_parts[7]),
        ],
        turning_point=[
            float(cleaned_parts[8]),
            float(cleaned_parts[9]),
            float(cleaned_parts[10]),
        ],
        out_location=[
            float(cleaned_parts[11]),
            float(cleaned_parts[12]),
            float(cleaned_parts[13]),
        ],
    )


def main() -> None:
    """Main processing function."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting data processing")

    validate_file(RAW_DATA_PATH, EXPECTED_SHA256)

    logger.info(f"Reading raw data from {RAW_DATA_PATH}")
    data_entries = []
    with open(RAW_DATA_PATH) as f:
        lines = f.readlines()[4:]  # Skip the first 4 header lines
        logger.info(f"Parsing {len(lines)} data entries")
        for line in lines:
            entry = parse_line(line)
            data_entries.append(entry)

    logger.info(f"Successfully parsed {len(data_entries)} entries")
    df = pd.DataFrame([asdict(entry) for entry in data_entries])
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
