"""
MotionSense データセット前処理

MotionSense データセット:
- 6種類の活動（Downstairs, Upstairs, Walking, Jogging, Standing, Sitting）
- 24名の被験者（性別・年齢・体重・身長の多様性）
- iPhone 6s（前ポケット装着）
- サンプリングレート: 50Hz
- DeviceMotion: attitude(3) + gravity(3) + rotationRate(3) + userAcceleration(3) = 12チャンネル
- 参照: https://github.com/mmalekzadeh/motion-sense
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from collections import defaultdict

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
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# MotionSense データセットのURL
MOTIONSENSE_URL = "https://github.com/mmalekzadeh/motion-sense/archive/refs/heads/master.zip"


@register_preprocessor('motionsense')
class MotionSensePreprocessor(BasePreprocessor):
    """
    MotionSense データセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # MotionSense固有の設定
        self.num_activities = 6
        self.num_subjects = 24
        self.num_sensors = 1  # Pocket (front pocket)

        # 活動とディレクトリ名のマッピング
        self.activity_map = {
            'dws': 0,  # Downstairs
            'ups': 1,  # Upstairs
            'wlk': 2,  # Walking
            'jog': 3,  # Jogging
            'std': 4,  # Standing
            'sit': 5   # Sitting
        }

        # トライアルコード（GitHub README.mdより）
        self.trial_codes = {
            'dws': [1, 2, 11],
            'ups': [3, 4, 12],
            'wlk': [7, 8, 15],
            'jog': [9, 16],
            'std': [6, 14],
            'sit': [5, 13]
        }

        # センサー名（位置: Pocket）
        self.sensor_names = ['Pocket']

        # DeviceMotionデータのカラム（12チャンネル）
        self.device_motion_columns = [
            'attitude.roll', 'attitude.pitch', 'attitude.yaw',      # ATT (3)
            'gravity.x', 'gravity.y', 'gravity.z',                  # GRA (3)
            'rotationRate.x', 'rotationRate.y', 'rotationRate.z',  # ROT (3)
            'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'  # ACC (3)
        ]

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'Pocket': {
                'ATT': (0, 3),   # attitude: roll, pitch, yaw
                'GRA': (3, 6),   # gravity: x, y, z
                'ROT': (6, 9),   # rotationRate: x, y, z
                'ACC': (9, 12),  # userAcceleration: x, y, z
            }
        }

        # サンプリングレート
        self.original_sampling_rate = 50  # Hz
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（dataset_info.pyから取得）
        self.scale_factor = DATASETS.get('MOTIONSENSE', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'motionsense'

    def download_dataset(self) -> None:
        """
        MotionSenseデータセットをダウンロードして解凍

        注: GitHubからダウンロードする場合は手動が推奨
        """
        raise NotImplementedError(
            "Please manually download MotionSense dataset from:\n"
            "https://github.com/mmalekzadeh/motion-sense\n"
            "Extract 'A_DeviceMotion_data' folder to: data/raw/motionsense/"
        )

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        全データファイルを読み込み、被験者単位にグループ化

        Returns:
            {subject_id: (data, labels)} の辞書
                data: (num_samples, 12) - DeviceMotionの12チャンネル
                labels: (num_samples,) - 活動ラベル
        """
        ms_dir = self.raw_data_path / 'motionsense' / 'A_DeviceMotion_data'

        if not ms_dir.exists():
            raise FileNotFoundError(
                f"MotionSense data not found at {ms_dir}\n"
                "Please run download_dataset() or manually download the dataset."
            )

        logger.info(f"Loading MotionSense data from {ms_dir}")

        # 被験者ごとにデータをグループ化
        subject_data_list = defaultdict(list)

        # 各活動について処理
        for act_name, act_label in self.activity_map.items():
            trials = self.trial_codes[act_name]

            for trial in trials:
                # ディレクトリ名: {activity}_{trial} (例: dws_1, ups_3)
                trial_dir = ms_dir / f"{act_name}_{trial}"

                if not trial_dir.exists():
                    logger.warning(f"Trial directory not found: {trial_dir}")
                    continue

                # 被験者ごとのCSVファイルを取得
                csv_files = sorted(trial_dir.glob('sub_*.csv'))

                for csv_file in csv_files:
                    # ファイル名から被験者IDを取得: sub_1.csv -> 1
                    try:
                        subject_id = int(csv_file.stem.split('_')[1])
                    except (IndexError, ValueError):
                        logger.warning(f"Invalid filename format: {csv_file.name}")
                        continue

                    try:
                        # CSVを読み込み（'Unnamed: 0'列は削除）
                        df = pd.read_csv(csv_file)
                        if 'Unnamed: 0' in df.columns:
                            df = df.drop(['Unnamed: 0'], axis=1)

                        # DeviceMotionの12カラムを抽出
                        data = df[self.device_motion_columns].values  # (N, 12)

                        # 全サンプルに同じラベルを付与
                        labels = np.full(len(data), act_label, dtype=np.int32)

                        # 被験者ごとのリストに追加
                        subject_data_list[subject_id].append((data, labels))
                        logger.info(f"Loaded {csv_file.name}: data={data.shape}, label={act_label} ({act_name})")

                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {e}")

        # 同一被験者の複数トライアルを結合
        all_data = {}
        for subject_id, file_list in subject_data_list.items():
            # データとラベルを時系列で結合
            all_subject_data = np.vstack([d for d, l in file_list])
            all_subject_labels = np.concatenate([l for d, l in file_list])

            all_data[subject_id] = (all_subject_data, all_subject_labels)
            logger.info(f"USER{subject_id:05d}: combined data={all_subject_data.shape}, "
                       f"labels={all_subject_labels.shape}")

        logger.info(f"Loaded {len(all_data)} subjects successfully")
        return all_data

    def clean_data(
        self,
        data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニングとリサンプリング
        """
        cleaned = {}

        for subject_id, (subject_data, labels) in data.items():
            # NaN除去
            valid_mask = ~np.isnan(subject_data).any(axis=1)
            cleaned_data = subject_data[valid_mask]
            cleaned_labels = labels[valid_mask]

            # リサンプリング (50Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[subject_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{subject_id:05d} resampled: {resampled_data.shape}")
            else:
                cleaned[subject_id] = (cleaned_data, cleaned_labels)
                logger.info(f"USER{subject_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(
        self,
        data: Dict[int, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for subject_id, (subject_data, labels) in data.items():
            logger.info(f"Processing USER{subject_id:05d}")
            processed[subject_id] = {}

            # Pocketセンサー（12チャンネル全て）
            sensor_name = 'Pocket'
            sensor_data = subject_data  # (N, 12)

            # スライディングウィンドウ適用
            windowed_data, windowed_labels = create_sliding_windows(
                sensor_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            # 各モダリティに分割
            modalities = self.sensor_modalities[sensor_name]
            for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]  # (num_windows, window_size, 3)

                # スケーリング適用（ACCのみ、scale_factorが定義されている場合）
                # MotionSenseの場合、userAccelerationは既にG単位なのでscale_factor=None
                if modality_name == 'ACC' and self.scale_factor is not None:
                    modality_data = modality_data / self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                # 形状変換: (N, T, C) -> (N, C, T)
                modality_data = np.transpose(modality_data, (0, 2, 1))

                # float16に変換（メモリ効率化）
                modality_data = modality_data.astype(np.float16)

                sensor_modality_key = f"{sensor_name}/{modality_name}"
                processed[subject_id][sensor_modality_key] = {
                    'X': modality_data,
                    'Y': windowed_labels
                }

                logger.info(f"  {sensor_modality_key}: X.shape={modality_data.shape}, "
                           f"Y.shape={windowed_labels.shape}")

        return processed

    def save_processed_data(
        self,
        data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        処理済みデータを保存

        保存形式:
            data/processed/motionsense/USER00001/Pocket/ATT/X.npy, Y.npy
            data/processed/motionsense/USER00001/Pocket/GRA/X.npy, Y.npy
            data/processed/motionsense/USER00001/Pocket/ROT/X.npy, Y.npy
            data/processed/motionsense/USER00001/Pocket/ACC/X.npy, Y.npy
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
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'users': {}
        }

        for subject_id, sensor_modality_data in data.items():
            user_name = f"USER{subject_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                X = arrays['X']
                Y = arrays['Y']

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}")

            total_stats['users'][user_name] = user_stats

        # メタデータ保存（NumPy型をJSON互換に変換）
        metadata_path = base_path / 'metadata.json'

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

        with open(metadata_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")
