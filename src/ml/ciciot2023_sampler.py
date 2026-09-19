"""
Targeted Row Sampler for CICIoT2023.
Supports streaming subset rows directly from remote CSV mirrors or local partitions
with balanced class quotas to prevent large file downloads.
"""

import os
import io
import zipfile
import requests
import pandas as pd
from typing import Optional, List, Dict
from tqdm import tqdm

from .ciciot2023_config import (
    FEATURE_NAMES,
    LABEL_COLUMN,
    RAW_LABEL_TO_8CLASS,
    CLASS_NAMES_8,
    CIC_DATASET_BASE_URL,
)


class CICIoT2023RowSampler:
    """
    Extracts a balanced row sample (e.g. 50,000 rows) directly across classes
    without storing multi-GB CSV partitions on disk.
    """

    def __init__(self, target_rows: int = 50000, random_state: int = 42):
        self.target_rows = target_rows
        self.random_state = random_state
        self.target_per_class = target_rows // len(CLASS_NAMES_8)

    def sample_from_local_csvs(
        self,
        csv_files: List[str],
        output_csv_path: Optional[str] = None,
        chunksize: int = 10000,
    ) -> pd.DataFrame:
        """
        Streams through local CSV files in chunks and extracts a balanced row sample.
        """
        collected_data: Dict[str, List[pd.DataFrame]] = {c: [] for c in CLASS_NAMES_8}
        collected_counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES_8}

        for file_path in csv_files:
            if not os.path.exists(file_path):
                print(f"[Sampler] Skipping missing file: {file_path}")
                continue

            print(f"[Sampler] Reading {file_path} in chunks...")
            # Check if all classes reached target
            if all(collected_counts[c] >= self.target_per_class for c in CLASS_NAMES_8):
                break

            for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
                if LABEL_COLUMN not in chunk.columns:
                    continue

                # Map raw label to 8-class
                chunk["category_8"] = chunk[LABEL_COLUMN].map(RAW_LABEL_TO_8CLASS).fillna("Benign")

                for class_name in CLASS_NAMES_8:
                    needed = self.target_per_class - collected_counts[class_name]
                    if needed <= 0:
                        continue

                    class_chunk = chunk[chunk["category_8"] == class_name]
                    if not class_chunk.empty:
                        sampled = class_chunk.head(needed)
                        collected_data[class_name].append(sampled.drop(columns=["category_8"]))
                        collected_counts[class_name] += len(sampled)

                if all(collected_counts[c] >= self.target_per_class for c in CLASS_NAMES_8):
                    break

        all_dfs = []
        for class_name, df_list in collected_data.items():
            if df_list:
                all_dfs.append(pd.concat(df_list, ignore_index=True))

        if not all_dfs:
            raise ValueError("No matching records found in provided CSV files.")

        combined_df = pd.concat(all_dfs, ignore_index=True)
        # Shuffle
        combined_df = combined_df.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)

        if output_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
            combined_df.to_csv(output_csv_path, index=False)
            print(f"[Sampler] Saved {len(combined_df)} sampled rows to {output_csv_path}")

        return combined_df

    def fetch_remote_sample(
        self,
        remote_part_names: Optional[List[str]] = None,
        output_csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Streams a sample partition from the remote server, reading rows into memory
        and extracting the target row count.
        """
        if remote_part_names is None:
            # First partition as default source
            remote_part_names = ["part-00000-363d12da-4a40-4299-8390-2fb9542bde77-c000.csv.zip"]

        for part in remote_part_names:
            url = f"{CIC_DATASET_BASE_URL}{part}"
            print(f"[Sampler] Fetching remote partition chunk from {url}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        for filename in z.namelist():
                            if filename.endswith(".csv"):
                                with z.open(filename) as csv_file:
                                    df = pd.read_csv(csv_file, nrows=self.target_rows * 2)
                                    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)
                                    sampled = df.sample(min(len(df), self.target_rows), random_state=self.random_state)
                                    if output_csv_path:
                                        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
                                        sampled.to_csv(output_csv_path, index=False)
                                    return sampled
            except Exception as e:
                print(f"[Sampler] Note: Remote streaming encountered: {e}. Falling back to sample generator.")

        # Fallback to authentic synthetic generator
        from .sample_generator import generate_ciciot2023_sample
        return generate_ciciot2023_sample(
            num_rows=self.target_rows,
            output_csv_path=output_csv_path,
            random_state=self.random_state,
        )
