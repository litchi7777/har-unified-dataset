"""
LARa (Logistic Activity Recognition) データセット前処理

LARa データセット:
- 8種類の身体活動（物流倉庫作業）
- 8人の被験者（S07-S14）
- 5つのセンサー（左腕、左脚、首、右腕、右脚）
- サンプリングレート: 100Hz
- 加速度センサー（3軸、G単位）+ ジャイロスコープ（3軸）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import json

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

# LARA データセットのURL（手動ダウンロードが必要）
LARA_URL = None  # 手動ダウンロードのみ


@register_preprocessor('lara')
class LaraPreprocessor(BasePreprocessor):
    """
    LARAデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # LARA固有の設定
        self.num_activities = 8
        self.num_subjects = 8  # S07-S14
        self.num_sensors = 5
        self.num_channels = 30  # 5センサー × (3軸ACC + 3軸GYRO)

        # センサー名とチャンネルマッピング
        # チャンネル構成:
        # LeftArm: ACC(3) + GYRO(3) = 6
        # LeftLeg: ACC(3) + GYRO(3) = 6
        # Neck: ACC(3) + GYRO(3) = 6
        # RightArm: ACC(3) + GYRO(3) = 6
        # RightLeg: ACC(3) + GYRO(3) = 6
        self.sensor_names = ['LeftArm', 'LeftLeg', 'Neck', 'RightArm', 'RightLeg']
        self.sensor_channel_ranges = {
            'LeftArm': (0, 6),      # channels 0-5
            'LeftLeg': (6, 12),     # channels 6-11
            'Neck': (12, 18),       # channels 12-17
            'RightArm': (18, 24),   # channels 18-23
            'RightLeg': (24, 30)    # channels 24-29
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'LeftArm': {
                'ACC': (0, 3),   # 3軸加速度
                'GYRO': (3, 6)   # 3軸ジャイロ
            },
            'LeftLeg': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            },
            'Neck': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            },
            'RightArm': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            },
            'RightLeg': {
                'ACC': (0, 3),
                'GYRO': (3, 6)
            }
        }

        # サンプリングレート
        self.original_sampling_rate = 100  # Hz (LARAのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（既にG単位なので不要）
        self.scale_factor = DATASETS.get('LARA', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'lara'

    def download_dataset(self) -> None:
        """
        LARAデータセットのダウンロード（手動ダウンロードのみサポート）

        既に解凍済みでZIPファイルが残っている場合は削除を提案
        """
        logger.info("=" * 80)
        logger.info("LARA dataset setup")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name

        # データディレクトリが存在するかチェック
        imu_dir = dataset_path / "IMU data (annotated) _ MbientLab"
        data_exists = imu_dir.exists() and any(imu_dir.rglob('*.csv'))

        # ZIPファイルが存在するかチェック
        zip_path = dataset_path / "lara_imu.zip"
        if zip_path.exists() and data_exists:
            logger.warning(f"LARA data is already extracted at {dataset_path}")
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
            logger.info(f"LARA data already exists at {dataset_path}")
            return

        # 手動ダウンロードの案内
        logger.info("LARA dataset requires manual download:")
        logger.info("  1. Visit: https://zenodo.org/record/3862782")
        logger.info("  2. Download: lara_imu.zip")
        logger.info(f"  3. Extract to: {dataset_path}/")
        logger.info("  Expected structure: data/raw/lara/IMU data (annotated) _ MbientLab/S07/...")
        logger.info("=" * 80)
        raise NotImplementedError(
            "LARA dataset must be downloaded manually.\n"
            "Visit: https://zenodo.org/record/3862782"
        )

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        LARAの生データを読み込む

        Returns:
            {person_id: (sensor_data, labels)} の辞書
            - sensor_data: (T, 30) 形状の配列（T=時系列長、30チャンネル）
            - labels: (T,) 形状の配列
        """
        logger.info(f"Loading LARA data from {self.raw_data_path}")

        # IMU data (annotated) _ MbientLab ディレクトリ
        imu_dir = self.raw_data_path / self.dataset_name / "IMU data (annotated) _ MbientLab"
        if not imu_dir.exists():
            raise FileNotFoundError(f"IMU data directory not found: {imu_dir}")

        # 被験者ディレクトリ（S07-S14）
        subject_dirs = sorted([d for d in imu_dir.iterdir() if d.is_dir() and d.name.startswith('S')])
        logger.info(f"Found {len(subject_dirs)} subjects: {[d.name for d in subject_dirs]}")

        result = {}
        person_id = 1  # 1-indexed

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name  # e.g., 'S07'
            logger.info(f"Processing subject {subject_id}...")

            # 各被験者のすべてのセッションファイルを取得
            data_files = sorted([f for f in subject_dir.glob('*.csv') if not f.name.endswith('_labels.csv')])
            logger.info(f"  Found {len(data_files)} sessions for {subject_id}")

            # 全セッションのデータを結合
            all_sensor_data = []
            all_labels = []

            for data_file in data_files:
                # 対応するラベルファイル
                label_file = data_file.parent / f"{data_file.stem}_labels.csv"
                if not label_file.exists():
                    logger.warning(f"  Label file not found for {data_file.name}, skipping...")
                    continue

                # データ読み込み
                df_data = pd.read_csv(data_file)
                df_labels = pd.read_csv(label_file)

                # Timeカラムを除外してセンサーデータを抽出
                # センサーデータの順序を統一
                # LA (LeftArm), LL (LeftLeg), N (Neck), RA (RightArm), RL (RightLeg)
                # 各センサー: AccelerometerX,Y,Z, GyroscopeX,Y,Z
                ordered_columns = []
                for sensor_prefix in ['LA', 'LL', 'N', 'RA', 'RL']:
                    for measurement in ['Accelerometer', 'Gyroscope']:
                        for axis in ['X', 'Y', 'Z']:
                            col_name = f"{sensor_prefix}_{measurement}{axis}"
                            ordered_columns.append(col_name)

                sensor_data = df_data[ordered_columns].values  # (T, 30)

                # ラベル抽出（Classカラム）
                labels = df_labels['Class'].values  # (T,)

                all_sensor_data.append(sensor_data)
                all_labels.append(labels)

            if not all_sensor_data:
                logger.warning(f"  No valid data for {subject_id}, skipping...")
                continue

            # 全セッションを時系列方向に結合
            combined_sensor_data = np.concatenate(all_sensor_data, axis=0)  # (T_total, 30)
            combined_labels = np.concatenate(all_labels, axis=0)  # (T_total,)

            logger.info(f"  Subject {subject_id}: {combined_sensor_data.shape[0]} samples")

            result[person_id] = (combined_sensor_data, combined_labels)
            person_id += 1

        logger.info(f"Loaded data for {len(result)} subjects")
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
            # リサンプリング (100Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    person_data,
                    labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[person_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{person_id:05d} resampled: {resampled_data.shape}")
            else:
                cleaned[person_id] = (person_data, labels)
                logger.info(f"USER{person_id:05d}: {person_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出とウィンドウ分割

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
        """
        result = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Extracting features for USER{person_id:05d}")

            # スライディングウィンドウ
            X, Y = create_sliding_windows(
                person_data,
                labels,
                window_size=self.window_size,
                stride=self.stride
            )
            # X shape: (num_windows, 30, window_size)
            # Y shape: (num_windows,)

            logger.info(f"  Generated {len(X)} windows")

            # センサーとモダリティごとに分割
            person_dict = {}

            for sensor_name in self.sensor_names:
                # チャンネル範囲を取得
                start_ch, end_ch = self.sensor_channel_ranges[sensor_name]

                # センサーのデータを抽出
                X_sensor = X[:, start_ch:end_ch, :]  # (num_windows, 6, window_size)

                # モダリティごとに分割
                for modality_name, (mod_start, mod_end) in self.sensor_modalities[sensor_name].items():
                    X_modality = X_sensor[:, mod_start:mod_end, :]  # (num_windows, 3, window_size)

                    sensor_modality_key = f"{sensor_name}/{modality_name}"
                    person_dict[sensor_modality_key] = {
                        'X': X_modality.astype(np.float16),
                        'Y': Y.astype(np.int32)
                    }

                    logger.info(f"  {sensor_modality_key}: X={X_modality.shape}, Y={Y.shape}")

            result[person_id] = person_dict

        return result

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        処理済みデータを保存

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        保存形式:
            data/processed/lara/USER00001/LeftArm/ACC/X.npy, Y.npy
            data/processed/lara/USER00001/LeftArm/GYRO/X.npy, Y.npy
            ...
        """
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

                # 統計情報を記録
                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'num_windows': len(X),
                    'shape': list(X.shape),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"  Saved {user_name}/{sensor_modality_name}: X={X.shape}, Y={Y.shape}")

            total_stats['users'][user_name] = user_stats

        # 統計情報をJSONで保存
        stats_path = base_path / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(total_stats, f, indent=2)

        logger.info(f"Saved dataset statistics to {stats_path}")

    def process(self) -> None:
        """
        LARAデータセットの前処理を実行
        """
        logger.info("="*80)
        logger.info("Starting LARA dataset preprocessing")
        logger.info("="*80)

        # 生データ読み込み
        raw_data = self.load_raw_data()

        # データクリーニングとリサンプリング
        cleaned_data = self.clean_data(raw_data)

        # 特徴抽出とウィンドウ分割
        features = self.extract_features(cleaned_data)

        # 保存
        self.save_processed_data(features)

        logger.info("="*80)
        logger.info("LARA preprocessing completed")
        logger.info("="*80)
