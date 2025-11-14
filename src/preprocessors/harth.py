"""
HARTH データセット前処理

HARTH データセット:
- 12種類の身体活動（サイクリング含む）
- 22人の被験者
- 2つのセンサー（腰部、右大腿部）
- サンプリングレート: 50Hz
- 加速度センサーのみ（3軸、G単位）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from .common import (
    download_file,
    extract_archive,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)

# HARTH データセットのURL（手動ダウンロードが必要）
HARTH_URL = None  # 手動ダウンロードのみ


@register_preprocessor('harth')
class HarthPreprocessor(BasePreprocessor):
    """
    HARTHデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # HARTH固有の設定
        self.num_activities = 12
        self.num_subjects = 22
        self.num_sensors = 2
        self.num_channels = 6  # 2センサー × 3軸

        # センサー名とチャンネルマッピング
        # チャンネル構成:
        # LowerBack: ACC(3) = 3
        # RightThigh: ACC(3) = 3
        self.sensor_names = ['LowerBack', 'RightThigh']
        self.sensor_channel_ranges = {
            'LowerBack': (0, 3),   # channels 0-2
            'RightThigh': (3, 6)   # channels 3-5
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'LowerBack': {
                'ACC': (0, 3)   # 3軸加速度
            },
            'RightThigh': {
                'ACC': (0, 3)   # 3軸加速度
            }
        }

        # サンプリングレート
        self.original_sampling_rate = 50  # Hz (HARTHのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（既にG単位なので不要）
        self.scale_factor = DATASETS.get('HARTH', {}).get('scale_factor', None)

        # ラベルマッピング（元の非連番ラベル → 0-indexed）
        # 1 -> 0 (Walking)
        # 2 -> 1 (Running)
        # 3 -> 2 (Shuffling)
        # 4 -> 3 (Stairs Up)
        # 5 -> 4 (Stairs Down)
        # 6 -> 5 (Standing)
        # 7 -> 6 (Sitting)
        # 8 -> 7 (Lying)
        # 13 -> 8 (Cycling Seated)
        # 14 -> 9 (Cycling Standing)
        # 130 -> 10 (Cycling Seated Inactive)
        # 140 -> 11 (Cycling Standing Inactive)
        self.label_mapping = {
            1: 0,    # Walking
            2: 1,    # Running
            3: 2,    # Shuffling
            4: 3,    # Stairs Up
            5: 4,    # Stairs Down
            6: 5,    # Standing
            7: 6,    # Sitting
            8: 7,    # Lying
            13: 8,   # Cycling Seated
            14: 9,   # Cycling Standing
            130: 10, # Cycling Seated Inactive
            140: 11  # Cycling Standing Inactive
        }

    def get_dataset_name(self) -> str:
        return 'harth'

    def download_dataset(self) -> None:
        """
        HARTHデータセットのダウンロード（手動ダウンロードのみサポート）

        既に解凍済みでZIPファイルが残っている場合は削除を提案
        """
        logger.info("=" * 80)
        logger.info("HARTH dataset setup")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name

        # データディレクトリが存在するかチェック
        data_dir = dataset_path / self.dataset_name
        data_exists = data_dir.exists() and any(data_dir.glob('S*.csv'))

        # ZIPファイルが存在するかチェック
        zip_path = dataset_path / "harth.zip"
        if zip_path.exists() and data_exists:
            logger.warning(f"HARTH data is already extracted at {dataset_path}")
            logger.warning(f"ZIP file still exists: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
            try:
                response = input("Do you want to delete the ZIP file? (Y/n): ")
                if response.lower() != 'n':
                    logger.info(f"Deleting {zip_path}")
                    zip_path.unlink()
                    logger.info("ZIP file deleted successfully")
                else:
                    logger.info("Keeping ZIP file")
            except EOFError:
                # 非対話環境の場合はデフォルトでYes
                logger.info(f"Non-interactive mode: Deleting {zip_path}")
                zip_path.unlink()
                logger.info("ZIP file deleted successfully")
            return

        # データセットが存在するかチェック（ZIPなし）
        if data_exists:
            logger.info(f"HARTH data already exists at {dataset_path}")
            return

        # 手動ダウンロードの案内
        logger.info("HARTH dataset requires manual download:")
        logger.info("  1. Visit: https://archive.ics.uci.edu/dataset/779/harth")
        logger.info("  2. Download: harth.zip")
        logger.info(f"  3. Extract to: {dataset_path}/")
        logger.info("  Expected structure: data/raw/harth/harth/S006.csv, ...")
        logger.info("=" * 80)
        raise NotImplementedError(
            "HARTH dataset must be downloaded manually.\n"
            "Visit: https://archive.ics.uci.edu/dataset/779/harth"
        )

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        HARTHの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/harth/harth/S006.csv
        - 各ファイル: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 6) の配列 [back_xyz, thigh_xyz]
                labels: (num_samples,) の配列（0-indexed）
        """
        raw_path = self.raw_data_path / self.dataset_name / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"HARTH raw data not found at {raw_path}\n"
                "Expected structure: data/raw/harth/harth/S006.csv"
            )

        # 被験者ごとにデータを格納
        result = {}

        # 利用可能なCSVファイルを検索
        csv_files = sorted(raw_path.glob('S*.csv'))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path}")

        # person_idは1-indexedで管理（USER00001から開始）
        for idx, subject_file in enumerate(csv_files):
            try:
                # CSVデータ読み込み
                df = pd.read_csv(subject_file)

                # カラム: timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
                if len(df.columns) != 8:
                    logger.warning(
                        f"Unexpected number of columns in {subject_file}: "
                        f"{len(df.columns)} (expected 8)"
                    )
                    continue

                # センサーデータ抽出（back_xyz + thigh_xyz）
                sensor_data = df.iloc[:, 1:7].values.astype(np.float32)
                # sensor_data: (num_samples, 6)

                # ラベル抽出とマッピング
                original_labels = df.iloc[:, 7].values.astype(int)
                labels = np.array([self.label_mapping[l] for l in original_labels], dtype=int)

                # person_idを1-indexedに変換（idx=0 -> person_id=1 -> USER00001）
                person_id = idx + 1
                result[person_id] = (sensor_data, labels)
                logger.info(
                    f"USER{person_id:05d} ({subject_file.name}): "
                    f"{sensor_data.shape}, Labels: {labels.shape}"
                )

            except Exception as e:
                logger.error(f"Error loading {subject_file}: {e}")
                continue

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニングとリサンプリング

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            クリーニング・リサンプリング済み {person_id: (data, labels)}
        """
        cleaned = {}
        for person_id, (person_data, labels) in data.items():
            # 無効なサンプルを除去
            cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)

            # リサンプリング (50Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[person_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{person_id:05d} cleaned and resampled: {resampled_data.shape}")
            else:
                cleaned[person_id] = (cleaned_data, cleaned_labels)
                logger.info(f"USER{person_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化）

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            例: {'LowerBack/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # センサーのチャンネルを抽出
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]

                # スライディングウィンドウ適用（最後のウィンドウはパディング）
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # HARTH: 150に満たない場合はパディング
                )
                # windowed_data: (num_windows, window_size, sensor_channels)

                # 各モダリティに分割（HARTHはACCのみ）
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # スケーリングは不要（既にG単位）
                    if self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # 形状を変換: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

                    # float16に変換
                    modality_data = modality_data.astype(np.float16)

                    # センサー/モダリティの階層構造
                    sensor_modality_key = f"{sensor_name}/{modality_name}"

                    processed[person_id][sensor_modality_key] = {
                        'X': modality_data,
                        'Y': windowed_labels
                    }

                    logger.info(
                        f"  {sensor_modality_key}: X.shape={modality_data.shape}, "
                        f"Y.shape={windowed_labels.shape}"
                    )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        処理済みデータを保存

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        保存形式:
            data/processed/harth/USER00001/LowerBack/ACC/X.npy, Y.npy
            data/processed/harth/USER00001/RightThigh/ACC/X.npy, Y.npy
            ...
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_sensors': self.num_sensors,
            'sensor_names': self.sensor_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # 正規化なし（生データ保持）
            'scale_factor': self.scale_factor,  # スケーリング係数（既にG単位なのでNone）
            'data_dtype': 'float16',  # データ型
            'users': {}
        }

        for person_id, sensor_modality_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # X.npy, Y.npy を保存
                X = arrays['X']  # (num_windows, C, window_size)
                Y = arrays['Y']  # (num_windows,)

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                # 統計情報
                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(
                    f"Saved {user_name}/{sensor_modality_name}: "
                    f"X{X.shape}, Y{Y.shape}"
                )

            total_stats['users'][user_name] = user_stats

        # 全体のメタデータを保存
        metadata_path = base_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            # NumPy型をJSON互換に変換
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, tuple):
                    return list(obj)
                return obj

            def recursive_convert(d):
                if isinstance(d, dict):
                    return {k: recursive_convert(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [recursive_convert(v) for v in d]
                else:
                    return convert_to_serializable(d)

            serializable_stats = recursive_convert(total_stats)
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")
