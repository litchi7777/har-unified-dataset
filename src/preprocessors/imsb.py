"""
IMSB (IM-SportingBehaviors) データセット前処理

IMSB データセット:
- 6種類のスポーツ行動（Badminton, Basketball, Cycling, Football, Skipping, Table Tennis）
- 20名の被験者（プロ＋アマチュアアスリート、20-30歳）
- 2つのセンサー（Wrist, Neck）※Thighは欠損値多いため除外
- サンプリングレート: 20Hz (推定)
- 参照: portals.au.edu.pk/imc
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
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# IMSB データセットのURL
IMSB_URL = "https://portals.au.edu.pk/imc/Content/dataset/IMSB%20dataset.zip"


@register_preprocessor('imsb')
class IMSBPreprocessor(BasePreprocessor):
    """
    IMSB (IM-SportingBehaviors) データセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # IMSB固有の設定
        self.num_activities = 6
        self.num_subjects = 20  # 実際は85ファイルだが、約20名
        self.num_sensors = 2  # WristとNeckのみ（Thighは除外）

        # 活動とディレクトリのマッピング
        self.activity_map = {
            'badminton': 0,
            'basketball': 1,
            'cycling': 2,
            'football': 3,
            'skipping': 4,
            'tabletennis': 5
        }

        # センサー名とCSVカラムのマッピング
        self.sensor_names = ['Wrist', 'Neck']
        self.sensor_columns = {
            'Wrist': ['wx', 'wy', 'wz'],  # 3軸加速度
            'Neck': ['nx', 'ny', 'nz']     # 3軸加速度
        }

        # モダリティ（各センサーはACCのみ）
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3),  # 3軸加速度
            },
            'Neck': {
                'ACC': (0, 3),  # 3軸加速度
            }
        }

        # サンプリングレート
        self.original_sampling_rate = 20  # Hz (推定値: 1000samples/50s)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（G単位、変換不要）
        self.scale_factor = DATASETS.get('IMSB', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'imsb'

    def download_dataset(self) -> None:
        """
        IMSBデータセットをダウンロードして解凍
        """
        zip_path = self.raw_data_path / 'imsb.zip'
        extract_dir = self.raw_data_path / 'imsb'

        # ダウンロード済みかチェック
        if extract_dir.exists():
            csv_files = list(extract_dir.rglob('*.csv'))
            if len(csv_files) >= 85:
                logger.info(f"IMSB data already exists at {extract_dir}")
                return

        # ダウンロード
        if not zip_path.exists():
            logger.info(f"Downloading IMSB dataset from {IMSB_URL}...")
            download_file(IMSB_URL, zip_path)
        else:
            logger.info(f"Using existing zip file: {zip_path}")

        # 解凍
        logger.info(f"Extracting IMSB data to {extract_dir}...")
        extract_archive(zip_path, self.raw_data_path / 'imsb')

        logger.info(f"Extraction completed: {extract_dir}")

    def parse_filename(self, filename: str) -> Tuple[int, str, str]:
        """
        ファイル名から情報を抽出

        ファイル名形式: 001-M-badminton.csv
        - 001: 被験者番号
        - M: 性別 (M=male, F=female)
        - badminton: 活動名

        Returns:
            (subject_id, gender, activity)
        """
        stem = Path(filename).stem
        parts = stem.split('-')

        if len(parts) >= 3:
            subject_id = int(parts[0])
            gender = parts[1]
            activity = parts[2]
            return subject_id, gender, activity
        else:
            raise ValueError(f"Invalid filename format: {filename}")

    def load_csv_file(self, csv_file: Path) -> Tuple[np.ndarray, int]:
        """
        CSVファイルを読み込み

        Returns:
            data: (num_samples, 6) - Wrist(3) + Neck(3)
            label: 活動ラベルID
        """
        # ファイル名から活動を取得
        _, _, activity = self.parse_filename(csv_file.name)
        label = self.activity_map.get(activity.lower())

        if label is None:
            raise ValueError(f"Unknown activity: {activity}")

        # CSVを読み込み
        df = pd.read_csv(csv_file)

        # 必要なカラムを抽出（Wrist + Neck、Thighは除外）
        wrist_cols = self.sensor_columns['Wrist']
        neck_cols = self.sensor_columns['Neck']

        # データを結合
        wrist_data = df[wrist_cols].values  # (N, 3)
        neck_data = df[neck_cols].values    # (N, 3)

        # 欠損値を除去
        valid_mask = ~(np.isnan(wrist_data).any(axis=1) | np.isnan(neck_data).any(axis=1))
        wrist_data = wrist_data[valid_mask]
        neck_data = neck_data[valid_mask]

        # 結合: (N, 6)
        data = np.hstack([wrist_data, neck_data])

        return data, label

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        全データファイルを読み込み、被験者単位にグループ化

        Returns:
            {subject_id: (data, labels)} の辞書
                data: (num_samples, 6) - Wrist(3) + Neck(3)
                labels: (num_samples,) - 活動ラベル
        """
        imsb_dir = self.raw_data_path / 'imsb'

        # 全CSVファイルを取得（活動ディレクトリ配下）
        csv_files = []
        for activity_dir in imsb_dir.iterdir():
            if activity_dir.is_dir() and not activity_dir.name.startswith('.'):
                csv_files.extend(activity_dir.glob('*.csv'))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {imsb_dir}\n"
                "Please run download_dataset() first."
            )

        logger.info(f"Loading {len(csv_files)} CSV files...")

        # 被験者ごとにファイルをグループ化
        from collections import defaultdict
        subject_data_list = defaultdict(list)

        for csv_file in csv_files:
            try:
                data, label = self.load_csv_file(csv_file)

                # ファイル名から被験者IDを取得
                subject_id, _, _ = self.parse_filename(csv_file.name)

                # 全サンプルに同じラベルを付与
                labels = np.full(len(data), label, dtype=np.int32)

                # 被験者ごとのリストに追加
                subject_data_list[subject_id].append((data, labels))
                logger.info(f"Loaded {csv_file.name}: data={data.shape}, label={label}")

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")

        # 同一被験者の複数ファイルを結合
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
            # NaN除去（既にload時に実施済みだが念のため）
            valid_mask = ~np.isnan(subject_data).any(axis=1)
            cleaned_data = subject_data[valid_mask]
            cleaned_labels = labels[valid_mask]

            # リサンプリング (20Hz -> 30Hz)
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
        特徴抽出（センサー×モダリティごとのウィンドウ化）

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for subject_id, (subject_data, labels) in data.items():
            logger.info(f"Processing USER{subject_id:05d}")
            processed[subject_id] = {}

            # Wrist と Neck に分割
            wrist_data = subject_data[:, 0:3]  # (N, 3)
            neck_data = subject_data[:, 3:6]   # (N, 3)

            for sensor_name, sensor_data in [('Wrist', wrist_data), ('Neck', neck_data)]:
                # スライディングウィンドウ適用
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )

                # ACCモダリティ
                modality_name = 'ACC'
                modality_data = windowed_data  # (num_windows, window_size, 3)

                # スケーリング適用（G単位なので不要だが、一応チェック）
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
            data/processed/imsb/USER00001/Wrist/ACC/X.npy, Y.npy
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
