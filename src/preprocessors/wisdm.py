"""
WISDM (Wireless Sensor Data Mining) スマートフォン & スマートウォッチ
アクティビティおよびバイオメトリクスデータセット前処理

WISDM データセット:
- 18種類の日常・スポーツ・食事・筆記動作
- 51人の被験者（ID: 1600-1650）
- 2デバイス × 2モダリティ（Phone/Watch × ACC/GYRO） = 12チャンネル
- サンプリングレート: 20Hz
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import register_preprocessor
from .base import BasePreprocessor
from .common import (
    check_dataset_exists,
    cleanup_temp_files,
    download_file,
    extract_archive,
)
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    get_class_distribution,
    resample_timeseries,
)
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

# UCI Machine Learning Repository から提供されるアーカイブ
WISDM_URL = (
    "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)

# アクティビティコードと名称（activity_key.txt の定義に準拠）
WISDM_ACTIVITY_CODES: List[Tuple[str, str]] = [
    ("A", "Walking"),
    ("B", "Jogging"),
    ("C", "Stairs"),
    ("D", "Sitting"),
    ("E", "Standing"),
    ("F", "Typing"),
    ("G", "BrushingTeeth"),
    ("H", "EatingSoup"),
    ("I", "EatingChips"),
    ("J", "EatingPasta"),
    ("K", "Drinking"),
    ("L", "EatingSandwich"),
    ("M", "Kicking"),
    ("O", "Catching"),
    ("P", "Dribbling"),
    ("Q", "Writing"),
    ("R", "Clapping"),
    ("S", "FoldingClothes"),
]

ACTIVITY_CODE_TO_ID: Dict[str, int] = {
    code: idx for idx, (code, _) in enumerate(WISDM_ACTIVITY_CODES)
}


@register_preprocessor("wisdm")
class WISDMPreprocessor(BasePreprocessor):
    """
    WISDMデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_subjects = 51
        self.num_activities = len(WISDM_ACTIVITY_CODES)
        self.device_names = ["Phone", "Watch"]
        self.modalities = ["ACC", "GYRO"]
        self.channels_per_modality = 3  # 3-axis

        self.original_sampling_rate = 20  # Hz
        self.target_sampling_rate = config.get("target_sampling_rate", 30)  # Hz

        # デフォルト: 5秒ウィンドウ & 1秒ストライド（target SRベース）
        default_window = int(self.target_sampling_rate * 5)
        default_stride = int(self.target_sampling_rate * 1)
        self.window_size = config.get("window_size", default_window)
        self.stride = config.get("stride", default_stride)

        self.scale_factor = DATASETS.get("WISDM", {}).get("scale_factor")

        # パス構造
        self.dataset_root_name = "wisdm-dataset"
        self.device_folder_map = {"Phone": "phone", "Watch": "watch"}
        self.modality_folder_map = {"ACC": "accel", "GYRO": "gyro"}

    def get_dataset_name(self) -> str:
        return "wisdm"

    # ------------------------------------------------------------------
    # ダウンロード & 整理
    # ------------------------------------------------------------------
    def download_dataset(self) -> None:
        """
        WISDMデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading WISDM dataset")
        logger.info("=" * 80)

        wisdm_raw_path = self.raw_data_path / self.dataset_name

        required_pattern = [
            f"{self.dataset_root_name}/raw/phone/accel/data_*_accel_phone.txt"
        ]
        if check_dataset_exists(wisdm_raw_path, required_files=required_pattern):
            logger.warning(f"WISDM data already exists at {wisdm_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != "y":
                logger.info("Skipping download")
                return

        try:
            wisdm_raw_path.mkdir(parents=True, exist_ok=True)
            zip_path = wisdm_raw_path.parent / "wisdm.zip"
            download_file(WISDM_URL, zip_path, desc="Downloading WISDM archive")

            extract_to = wisdm_raw_path.parent / "wisdm_temp"
            extract_archive(zip_path, extract_to, desc="Extracting WISDM archive")
            self._organize_wisdm_data(extract_to, wisdm_raw_path)

            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: WISDM dataset available at {wisdm_raw_path}")
            logger.info("=" * 80)
        except Exception as exc:
            logger.error(f"Failed to download WISDM dataset: {exc}", exc_info=True)
            raise

    def _organize_wisdm_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        ネストされたアーカイブを解凍し、data/raw/wisdm 以下に整理
        """
        logger.info(f"Organizing WISDM data from {extracted_path} to {target_path}")
        target_path.mkdir(parents=True, exist_ok=True)

        # ネスト済みZIP（wisdm-dataset.zip 等）を解凍
        nested_archives = list(extracted_path.glob("*.zip"))
        for nested_archive in nested_archives:
            logger.info(f"Found nested archive: {nested_archive.name}")
            nested_extract = extracted_path / nested_archive.stem
            extract_archive(nested_archive, nested_extract, desc="Extracting nested WISDM")
            nested_archive.unlink()

        dataset_root = self._locate_dataset_root(extracted_path)
        if dataset_root is None:
            raise FileNotFoundError(
                f"Unable to locate WISDM raw directory under {extracted_path}"
            )

        target_dataset_dir = target_path / dataset_root.name
        if target_dataset_dir.exists():
            shutil.rmtree(target_dataset_dir)

        shutil.copytree(dataset_root, target_dataset_dir)
        logger.info(f"Copied dataset contents to {target_dataset_dir}")

    def _locate_dataset_root(self, search_root: Path) -> Optional[Path]:
        """
        最終的な raw ディレクトリを含むベースフォルダを探索
        """
        candidate = search_root / self.dataset_root_name
        if (candidate / "raw").exists():
            return candidate

        # 再帰的に raw ディレクトリを探索
        for raw_dir in search_root.rglob("raw"):
            if (raw_dir / "phone").exists() and (raw_dir / "watch").exists():
                return raw_dir.parent

        return None

    # ------------------------------------------------------------------
    # データ読み込み
    # ------------------------------------------------------------------
    def load_raw_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        生データを被験者 × デバイス × モダリティごとに読み込む
        """
        dataset_root = self._find_existing_dataset_root()
        raw_dir = dataset_root / "raw"

        person_data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}

        for device_name in self.device_names:
            device_folder = self.device_folder_map[device_name]

            for modality_name in self.modalities:
                modality_folder = self.modality_folder_map[modality_name]
                sensor_dir = raw_dir / device_folder / modality_folder

                if not sensor_dir.exists():
                    logger.warning(f"Sensor directory missing: {sensor_dir}")
                    continue

                for sensor_file in sorted(sensor_dir.glob("data_*_*.txt")):
                    subject_id = self._parse_subject_id(sensor_file.name)
                    data, labels = self._load_sensor_file(sensor_file)

                    if data is None or labels is None or len(data) == 0:
                        logger.warning(f"No valid samples in {sensor_file}")
                        continue

                    person_entry = person_data.setdefault(subject_id, {})
                    device_entry = person_entry.setdefault(device_name, {})
                    device_entry[modality_name] = (data, labels)

                    logger.info(
                        f"Loaded {sensor_file.name}: subject={subject_id}, "
                        f"{device_name}/{modality_name} -> {data.shape}"
                    )

        if not person_data:
            raise ValueError("No WISDM data loaded. Please check the raw data directory.")

        logger.info(f"Total subjects loaded: {len(person_data)}")
        return person_data

    def _find_existing_dataset_root(self) -> Path:
        """
        data/raw/wisdm 以下で raw ディレクトリを探索して返す
        """
        base_path = self.raw_data_path / self.dataset_name
        candidates = [
            base_path / self.dataset_root_name,
            base_path,
        ]

        for candidate in candidates:
            raw_dir = candidate / "raw"
            if raw_dir.exists():
                return candidate

        for raw_dir in base_path.rglob("raw"):
            if (raw_dir / "phone").exists():
                return raw_dir.parent

        raise FileNotFoundError(
            f"Could not locate WISDM raw directory under {base_path}. "
            "Expected structure: data/raw/wisdm/wisdm-dataset/raw/phone/accel/*.txt"
        )

    @staticmethod
    def _parse_subject_id(filename: str) -> int:
        """
        data_<subject>_*_* 形式のファイル名から subject_id を抽出
        """
        try:
            return int(filename.split("_")[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid WISDM filename format: {filename}") from exc

    def _load_sensor_file(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        単一のセンサーファイルを読み込み、データとラベルを返す
        """
        samples: List[List[float]] = []
        labels: List[int] = []
        timestamps: List[int] = []

        with file_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(";"):
                    line = line[:-1]

                parts = line.split(",")
                if len(parts) != 6:
                    logger.debug(f"Skipping malformed line ({file_path.name}): {line}")
                    continue

                _, activity_code, timestamp_str, x_str, y_str, z_str = parts
                activity_code = activity_code.strip().upper()
                if activity_code not in ACTIVITY_CODE_TO_ID:
                    logger.debug(
                        f"Unknown activity code '{activity_code}' in {file_path.name}"
                    )
                    continue

                try:
                    timestamp = int(timestamp_str)
                    x = float(x_str)
                    y = float(y_str)
                    z = float(z_str)
                except ValueError:
                    logger.debug(f"Invalid numeric values in {file_path.name}: {line}")
                    continue

                samples.append([x, y, z])
                labels.append(ACTIVITY_CODE_TO_ID[activity_code])
                timestamps.append(timestamp)

        if not samples:
            return None, None

        data = np.asarray(samples, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        timestamp_array = np.asarray(timestamps, dtype=np.int64)

        # タイムスタンプでソートして確実に時間順にする
        order = np.argsort(timestamp_array)
        data = data[order]
        label_array = label_array[order]

        return data, label_array

    # ------------------------------------------------------------------
    # クリーニング & リサンプリング
    # ------------------------------------------------------------------
    def clean_data(
        self, data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        cleaned: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = {}

        for person_id, device_data in data.items():
            cleaned[person_id] = {}
            for device_name, modality_data in device_data.items():
                cleaned[person_id][device_name] = {}

                for modality_name, (sensor_data, labels) in modality_data.items():
                    filtered_data, filtered_labels = filter_invalid_samples(sensor_data, labels)

                    if len(filtered_data) == 0:
                        logger.warning(
                            f"No valid samples after cleaning: USER{person_id} {device_name}/{modality_name}"
                        )
                        continue

                    if self.original_sampling_rate != self.target_sampling_rate:
                        resampled_data, resampled_labels = resample_timeseries(
                            filtered_data,
                            filtered_labels,
                            self.original_sampling_rate,
                            self.target_sampling_rate,
                        )
                    else:
                        resampled_data, resampled_labels = filtered_data, filtered_labels

                    cleaned[person_id][device_name][modality_name] = (
                        resampled_data,
                        resampled_labels,
                    )

                    logger.info(
                        f"USER{person_id:05d} {device_name}/{modality_name}: "
                        f"{sensor_data.shape} -> {resampled_data.shape}"
                    )

        return cleaned

    # ------------------------------------------------------------------
    # 特徴抽出
    # ------------------------------------------------------------------
    def extract_features(
        self, data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        processed: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

        for person_id, device_data in data.items():
            processed[person_id] = {}
            logger.info(f"Processing USER{person_id:05d}")

            for device_name in self.device_names:
                modality_data = device_data.get(device_name, {})

                for modality_name in self.modalities:
                    sensor_entry = modality_data.get(modality_name)
                    if sensor_entry is None:
                        continue

                    sensor_data, labels = sensor_entry
                    if len(sensor_data) < 1:
                        continue

                    windows, window_labels = create_sliding_windows(
                        sensor_data,
                        labels,
                        window_size=self.window_size,
                        stride=self.stride,
                        drop_last=False,
                        pad_last=True,
                    )

                    if len(windows) == 0:
                        logger.warning(
                            f"No windows generated for USER{person_id:05d} "
                            f"{device_name}/{modality_name}"
                        )
                        continue

                    if modality_name == "ACC" and self.scale_factor:
                        windows = windows / self.scale_factor

                    windows = np.transpose(windows, (0, 2, 1))
                    windows = windows.astype(np.float16)

                    key = f"{device_name}/{modality_name}"
                    processed[person_id][key] = {"X": windows, "Y": window_labels}

                    logger.info(
                        f"  {key}: X{windows.shape}, Y{window_labels.shape}"
                    )

        return processed

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save_processed_data(
        self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            "dataset": self.dataset_name,
            "num_activities": self.num_activities,
            "num_devices": len(self.device_names),
            "device_names": self.device_names,
            "modalities": self.modalities,
            "channels_per_modality": self.channels_per_modality,
            "original_sampling_rate": self.original_sampling_rate,
            "target_sampling_rate": self.target_sampling_rate,
            "window_size": self.window_size,
            "stride": self.stride,
            "normalization": "none",
            "scale_factor": self.scale_factor,
            "data_dtype": "float16",
            "data_shape": f"(num_windows, {self.channels_per_modality}, {self.window_size})",
            "users": {},
        }

        for person_id, sensor_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {"sensor_modalities": {}}

            for sensor_modality_name, arrays in sensor_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                X = arrays["X"]
                Y = arrays["Y"]

                np.save(sensor_modality_path / "X.npy", X)
                np.save(sensor_modality_path / "Y.npy", Y)

                user_stats["sensor_modalities"][sensor_modality_name] = {
                    "X_shape": X.shape,
                    "Y_shape": Y.shape,
                    "num_windows": len(Y),
                    "class_distribution": get_class_distribution(Y),
                }

                logger.info(
                    f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}"
                )

            total_stats["users"][user_name] = user_stats

        metadata_path = base_path / "metadata.json"
        self._write_metadata(metadata_path, total_stats)
        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")

    def _write_metadata(self, metadata_path: Path, stats: Dict[str, Any]) -> None:
        """
        NumPy型を含む辞書をJSON保存可能な形式に変換して記録
        """
        import json

        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        def recursive(item):
            if isinstance(item, dict):
                return {k: recursive(v) for k, v in item.items()}
            if isinstance(item, list):
                return [recursive(v) for v in item]
            return convert(item)

        serializable_stats = recursive(stats)
        with open(metadata_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)
