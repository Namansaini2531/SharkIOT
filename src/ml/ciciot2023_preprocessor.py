"""
Preprocessing, Normalization, Label Encoding, and Splitting Pipeline for CICIoT2023.
Produces clean Train/Validation/Test splits for Binary and 8-Class Classification.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

from .ciciot2023_config import (
    FEATURE_NAMES,
    LABEL_COLUMN,
    RAW_LABEL_TO_8CLASS,
    CLASS_NAMES_8,
)


class CICIoT2023Preprocessor:
    """
    End-to-end preprocessing pipeline for CICIoT2023:
    1. Cleans columns and validates 46 numerical flow features.
    2. Imputes NaNs / infinite values using training set statistics.
    3. Derives Stage 1 (Binary: Benign=0, Attack=1) and Stage 2 (8-Class categorical) labels.
    4. Performs stratified Train/Val/Test splits.
    5. Fits Scaler (StandardScaler/MinMaxScaler) strictly on Train set.
    6. Serializes artifacts (scaler, encoders, preprocessed arrays, metadata) for training & live inference.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        self.scaler_type = scaler_type.lower()
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler_type '{scaler_type}'. Choose 'standard', 'minmax', or 'robust'.")

        self.label_encoder_8class = LabelEncoder()
        self.label_encoder_8class.fit(CLASS_NAMES_8)
        self.impute_values: Optional[np.ndarray] = None
        self.feature_names = list(FEATURE_NAMES)

    def clean_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cleans dataframe, validates 46 features, and extracts (X, y_binary, y_8class).
        """
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        # Check for label column
        if LABEL_COLUMN not in df.columns:
            # Fallback to last column if named differently
            raw_labels = df.iloc[:, -1].astype(str)
            feature_df = df.iloc[:, :-1]
        else:
            raw_labels = df[LABEL_COLUMN].astype(str)
            feature_df = df.drop(columns=[LABEL_COLUMN])

        # Align features
        missing_feats = [f for f in self.feature_names if f not in feature_df.columns]
        if missing_feats:
            print(f"[Preprocessor] Warning: Missing {len(missing_feats)} expected features. Filling with 0.0")
            for f in missing_feats:
                feature_df[f] = 0.0

        # Keep strictly the 46 features in canonical order
        feature_df = feature_df[self.feature_names]

        # Convert to float32 numpy array
        X_raw = feature_df.to_numpy(dtype=np.float32)

        # Replace inf and -inf with nan
        X_raw[np.isinf(X_raw)] = np.nan

        # Map labels to 8-class categories
        mapped_8class = raw_labels.map(RAW_LABEL_TO_8CLASS).fillna("Benign").values
        y_8class = self.label_encoder_8class.transform(mapped_8class)

        # Derive Binary labels: 0 for Benign, 1 for Attack
        y_binary = np.where(mapped_8class == "Benign", 0, 1).astype(np.int64)

        return X_raw, y_binary, y_8class

    def fit_transform(
        self,
        df: pd.DataFrame,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: Clean -> Split -> Impute -> Scale -> Save.
        """
        X_raw, y_binary, y_8class = self.clean_dataframe(df)

        print(f"[Preprocessor] Total dataset size: {len(X_raw)} rows, {X_raw.shape[1]} features.")
        print(f"[Preprocessor] Binary distribution -> Benign (0): {np.sum(y_binary == 0)}, Malicious (1): {np.sum(y_binary == 1)}")

        # Step 1: Train / (Val + Test) split with stratification on 8-class target
        total_eval_size = self.test_size + self.val_size
        X_train_raw, X_temp_raw, y_bin_train, y_bin_temp, y_8_train, y_8_temp = train_test_split(
            X_raw,
            y_binary,
            y_8class,
            test_size=total_eval_size,
            random_state=self.random_state,
            stratify=y_8class,
        )

        # Step 2: Val / Test split
        if self.val_size > 0:
            val_ratio = self.val_size / total_eval_size
            X_val_raw, X_test_raw, y_bin_val, y_bin_test, y_8_val, y_8_test = train_test_split(
                X_temp_raw,
                y_bin_temp,
                y_8_temp,
                test_size=(1.0 - val_ratio),
                random_state=self.random_state,
                stratify=y_8_temp,
            )
        else:
            X_val_raw = np.empty((0, X_raw.shape[1]), dtype=np.float32)
            y_bin_val = np.empty(0, dtype=np.int64)
            y_8_val = np.empty(0, dtype=np.int64)
            X_test_raw = X_temp_raw
            y_bin_test = y_bin_temp
            y_8_test = y_8_temp

        # Step 3: Compute median imputations strictly from Training set
        self.impute_values = np.nanmedian(X_train_raw, axis=0)
        # Handle cases where all values in column might be NaN
        self.impute_values = np.nan_to_num(self.impute_values, nan=0.0)

        # Impute NaNs across splits
        def impute(X_arr):
            X_copy = X_arr.copy()
            nan_mask = np.isnan(X_copy)
            if np.any(nan_mask):
                inds = np.where(nan_mask)
                X_copy[inds] = np.take(self.impute_values, inds[1])
            return X_copy

        X_train_clean = impute(X_train_raw)
        X_val_clean = impute(X_val_raw) if len(X_val_raw) > 0 else X_val_raw
        X_test_clean = impute(X_test_raw)

        # Step 4: Fit Scaler strictly on X_train_clean
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        X_val_scaled = self.scaler.transform(X_val_clean) if len(X_val_clean) > 0 else X_val_clean
        X_test_scaled = self.scaler.transform(X_test_clean)

        processed_bundle = {
            "X_train": X_train_scaled,
            "y_binary_train": y_bin_train,
            "y_8class_train": y_8_train,
            "X_val": X_val_scaled,
            "y_binary_val": y_bin_val,
            "y_8class_val": y_8_val,
            "X_test": X_test_scaled,
            "y_binary_test": y_bin_test,
            "y_8class_test": y_8_test,
            "feature_names": self.feature_names,
            "class_names_8": list(self.label_encoder_8class.classes_),
        }

        print(f"[Preprocessor] Splits completed:")
        print(f"  - Train: {len(X_train_scaled)} rows ({len(X_train_scaled)/len(X_raw)*100:.1f}%)")
        print(f"  - Val:   {len(X_val_scaled)} rows ({len(X_val_scaled)/len(X_raw)*100:.1f}%)")
        print(f"  - Test:  {len(X_test_scaled)} rows ({len(X_test_scaled)/len(X_raw)*100:.1f}%)")

        if output_dir:
            self.save_artifacts(output_dir, processed_bundle)

        return processed_bundle

    def save_artifacts(self, output_dir: str, bundle: Dict[str, Any]) -> None:
        """
        Saves processed numpy files, fitted transformers, and metadata.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save compressed numpy split arrays (for fast Python loading)
        np.savez_compressed(
            os.path.join(output_dir, "train.npz"),
            X=bundle["X_train"],
            y_binary=bundle["y_binary_train"],
            y_8class=bundle["y_8class_train"],
        )
        np.savez_compressed(
            os.path.join(output_dir, "val.npz"),
            X=bundle["X_val"],
            y_binary=bundle["y_binary_val"],
            y_8class=bundle["y_8class_val"],
        )
        np.savez_compressed(
            os.path.join(output_dir, "test.npz"),
            X=bundle["X_test"],
            y_binary=bundle["y_binary_test"],
            y_8class=bundle["y_8class_test"],
        )

        # 2. Save human-readable CSV files (for viewing, Excel, or custom inspection)
        def export_split_csv(X_data, y_bin, y_8, filename):
            df_export = pd.DataFrame(X_data, columns=self.feature_names)
            df_export["label_binary"] = y_bin
            df_export["label_binary_name"] = np.where(y_bin == 0, "Benign", "Malicious")
            df_export["label_8class"] = y_8
            df_export["label_8class_name"] = [bundle["class_names_8"][i] for i in y_8]
            df_export.to_csv(os.path.join(output_dir, filename), index=False)

        export_split_csv(bundle["X_train"], bundle["y_binary_train"], bundle["y_8class_train"], "train.csv")
        export_split_csv(bundle["X_val"], bundle["y_binary_val"], bundle["y_8class_val"], "val.csv")
        export_split_csv(bundle["X_test"], bundle["y_binary_test"], bundle["y_8class_test"], "test.csv")

        # 3. Save Scaler and Label Encoder
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.joblib"))
        joblib.dump(self.label_encoder_8class, os.path.join(output_dir, "label_encoder_8class.joblib"))
        joblib.dump(self.impute_values, os.path.join(output_dir, "impute_values.joblib"))


        # 3. Save Summary Metadata JSON
        class_dist_train = {
            cls: int(np.sum(bundle["y_8class_train"] == idx))
            for idx, cls in enumerate(bundle["class_names_8"])
        }
        metadata = {
            "dataset": "CICIoT2023",
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "scaler_type": self.scaler_type,
            "total_samples": len(bundle["X_train"]) + len(bundle["X_val"]) + len(bundle["X_test"]),
            "train_samples": len(bundle["X_train"]),
            "val_samples": len(bundle["X_val"]),
            "test_samples": len(bundle["X_test"]),
            "class_names_8": bundle["class_names_8"],
            "class_distribution_train_8class": class_dist_train,
            "binary_distribution_train": {
                "Benign (0)": int(np.sum(bundle["y_binary_train"] == 0)),
                "Malicious (1)": int(np.sum(bundle["y_binary_train"] == 1)),
            },
        }

        with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"[Preprocessor] Preprocessing artifacts saved to {output_dir}")

    def transform_single(self, raw_feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Transforms a single raw feature vector (or dictionary) for live inference in IoTShark.
        """
        vec = np.array([raw_feature_dict.get(f, 0.0) for f in self.feature_names], dtype=np.float32).reshape(1, -1)
        nan_mask = np.isnan(vec)
        if np.any(nan_mask) and self.impute_values is not None:
            inds = np.where(nan_mask)
            vec[inds] = np.take(self.impute_values, inds[1])
        return self.scaler.transform(vec)


def load_stage1_data(processed_dir: str = "data/ciciot2023/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads Stage 1 (Binary Classification: 0=Benign, 1=Malicious).

    Returns:
        X_train, y_binary_train, X_val, y_binary_val, X_test, y_binary_test
    """
    train = np.load(os.path.join(processed_dir, "train.npz"))
    val = np.load(os.path.join(processed_dir, "val.npz"))
    test = np.load(os.path.join(processed_dir, "test.npz"))
    return train["X"], train["y_binary"], val["X"], val["y_binary"], test["X"], test["y_binary"]


def load_stage2_data(processed_dir: str = "data/ciciot2023/processed") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Loads Stage 2 (8-Class Threat Classification).

    Returns:
        X_train, y_8class_train, X_val, y_8class_val, X_test, y_8class_test, class_names_8
    """
    train = np.load(os.path.join(processed_dir, "train.npz"))
    val = np.load(os.path.join(processed_dir, "val.npz"))
    test = np.load(os.path.join(processed_dir, "test.npz"))
    le = joblib.load(os.path.join(processed_dir, "label_encoder_8class.joblib"))
    return train["X"], train["y_8class"], val["X"], val["y_8class"], test["X"], test["y_8class"], list(le.classes_)

