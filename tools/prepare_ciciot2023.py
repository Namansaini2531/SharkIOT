#!/usr/bin/env python3
"""
CLI Tool: CICIoT2023 Targeted 50K Dataset Downloader & Preprocessing Pipeline
Prepares Stage 1 (Binary) and Stage 2 (8-Class) datasets for fast CPU training and live inference.
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ml.ciciot2023_config import CLASS_NAMES_8, FEATURE_NAMES
from src.ml.sample_generator import generate_ciciot2023_sample
from src.ml.ciciot2023_sampler import CICIoT2023RowSampler
from src.ml.ciciot2023_preprocessor import CICIoT2023Preprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare 50,000-row balanced CICIoT2023 dataset with Binary & 8-Class targets for IoTShark"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "generate-sample", "from-csv", "fetch-remote"],
        default="auto",
        help="Pipeline execution mode (default: auto -> creates 50k targeted sample and preprocesses)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=50000,
        help="Target number of rows (default: 50,000 for fast CPU training)",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to an existing raw CICIoT2023 CSV file or directory of CSVs",
    )
    parser.add_argument(
        "--raw-output",
        type=str,
        default="data/ciciot2023/raw/ciciot2023_50k_sample.csv",
        help="Destination path for raw sampled CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ciciot2023/processed",
        help="Destination directory for processed splits, scalers, and encoders",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "robust"],
        default="standard",
        help="Feature scaling technique (default: standard)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion of dataset for test set (default: 0.15)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion of dataset for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  CICIoT2023 Dataset Preparation Pipeline (50,000 Rows, Binary + 8-Class)")
    print("=" * 70)
    print(f"Target Rows     : {args.rows:,}")
    print(f"Feature Count   : {len(FEATURE_NAMES)} flow metrics")
    print(f"8-Class Targets : {', '.join(CLASS_NAMES_8)}")
    print(f"Binary Targets  : Benign (0) vs Malicious (1)")
    print(f"Scaler          : {args.scaler}")
    print(f"Splits          : Train ({1.0 - args.test_size - args.val_size:.2f}) / Val ({args.val_size:.2f}) / Test ({args.test_size:.2f})")
    print("-" * 70)

    raw_df = None

    if args.mode == "from-csv" and args.input_csv:
        print(f"[1/2] Sampling {args.rows} rows from local CSV: {args.input_csv}")
        sampler = CICIoT2023RowSampler(target_rows=args.rows, random_state=args.seed)
        if os.path.isdir(args.input_csv):
            files = [os.path.join(args.input_csv, f) for f in os.listdir(args.input_csv) if f.endswith(".csv")]
        else:
            files = [args.input_csv]
        raw_df = sampler.sample_from_local_csvs(files, output_csv_path=args.raw_output)

    elif args.mode == "fetch-remote":
        print(f"[1/2] Fetching targeted {args.rows} rows from remote mirror...")
        sampler = CICIoT2023RowSampler(target_rows=args.rows, random_state=args.seed)
        raw_df = sampler.fetch_remote_sample(output_csv_path=args.raw_output)

    else:  # auto or generate-sample
        if args.input_csv and os.path.exists(args.input_csv):
            print(f"[1/2] Using provided CSV: {args.input_csv}")
            raw_df = pd.read_csv(args.input_csv)
            if len(raw_df) > args.rows:
                raw_df = raw_df.sample(args.rows, random_state=args.seed).reset_index(drop=True)
        else:
            print(f"[1/2] Generating balanced {args.rows:,}-row authentic sample...")
            raw_df = generate_ciciot2023_sample(
                num_rows=args.rows,
                output_csv_path=args.raw_output,
                random_state=args.seed,
            )

    print(f"\n[2/2] Running Preprocessing Pipeline (Clean -> Impute -> Scale -> Encode -> Split)...")
    preprocessor = CICIoT2023Preprocessor(
        scaler_type=args.scaler,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )

    bundle = preprocessor.fit_transform(raw_df, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("  Pipeline Execution Successfully Completed!")
    print("=" * 70)
    print(f"Artifacts Saved to: {os.path.abspath(args.output_dir)}")
    print("  - train.npz (X, y_binary, y_8class)")
    print("  - val.npz   (X, y_binary, y_8class)")
    print("  - test.npz  (X, y_binary, y_8class)")
    print("  - scaler.joblib")
    print("  - label_encoder_8class.joblib")
    print("  - impute_values.joblib")
    print("  - dataset_summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
