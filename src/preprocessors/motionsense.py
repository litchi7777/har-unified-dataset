"""
MotionSense データセット前処理

MotionSense データセット:
- 6種類の活動（Downstairs, Upstairs, Walking, Jogging, Standing, Sitting）
- 24名の被験者（性別・年齢・体重・身長の多様性）
- iPhone 6s（前ポケット装着）
- サンプリングレート: 50Hz
- Accelerometer: x, y, z (3チャンネル、生の加速度）
- Gyroscope: x, y, z (3チャンネル）
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

        # Accelerometer と Gyroscope のカラム名
        self.acc_columns = ['x', 'y', 'z']
        self.gyro_columns = ['x', 'y', 'z']

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'Pocket': {
                'ACC': (0, 3),   # accelerometer: x, y, z
                'GYRO': (3, 6),  # gyroscope: x, y, z
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
        Accelerometer_data と Gyroscope_data を取得
        """
        import zipfile
        import tempfile
        import shutil

        ms_dir = self.raw_data_path / 'motionsense'
        acc_dir = ms_dir / 'B_Accelerometer_data'
        gyro_dir = ms_dir / 'C_Gyroscope_data'

        # 既にダウンロード済みかチェック
        if acc_dir.exists() and gyro_dir.exists():
            acc_files = list(acc_dir.glob('*/*.csv'))
            gyro_files = list(gyro_dir.glob('*/*.csv'))
            if len(acc_files) >= 100 and len(gyro_files) >= 100:
                logger.info(f"MotionSense data already exists at {ms_dir}")
                return

        ms_dir.mkdir(parents=True, exist_ok=True)

        # GitHubからクローン（一時ディレクトリ）
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            logger.info(f"Cloning MotionSense repository...")

            # git cloneを実行
            import subprocess
            result = subprocess.run(
                ['git', 'clone', '--depth=1',
                 'https://github.com/mmalekzadeh/motion-sense.git',
                 str(tmpdir / 'motion-sense')],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repository: {result.stderr}")

            # Accelerometer ZIPファイルを解凍
            acc_zip = tmpdir / 'motion-sense' / 'data' / 'B_Accelerometer_data.zip'
            if not acc_zip.exists():
                raise FileNotFoundError(f"Accelerometer ZIP not found: {acc_zip}")

            logger.info(f"Extracting {acc_zip.name}...")
            with zipfile.ZipFile(acc_zip, 'r') as zip_ref:
                zip_ref.extractall(tmpdir / 'motion-sense' / 'data')

            # Gyroscope ZIPファイルを解凍
            gyro_zip = tmpdir / 'motion-sense' / 'data' / 'C_Gyroscope_data.zip'
            if not gyro_zip.exists():
                raise FileNotFoundError(f"Gyroscope ZIP not found: {gyro_zip}")

            logger.info(f"Extracting {gyro_zip.name}...")
            with zipfile.ZipFile(gyro_zip, 'r') as zip_ref:
                zip_ref.extractall(tmpdir / 'motion-sense' / 'data')

            # フォルダを移動
            extracted_acc = tmpdir / 'motion-sense' / 'data' / 'B_Accelerometer_data'
            extracted_gyro = tmpdir / 'motion-sense' / 'data' / 'C_Gyroscope_data'

            if not extracted_acc.exists():
                raise FileNotFoundError(f"Extracted accelerometer dir not found: {extracted_acc}")
            if not extracted_gyro.exists():
                raise FileNotFoundError(f"Extracted gyroscope dir not found: {extracted_gyro}")

            # 既存のディレクトリがあれば削除
            if acc_dir.exists():
                shutil.rmtree(acc_dir)
            if gyro_dir.exists():
                shutil.rmtree(gyro_dir)

            shutil.move(str(extracted_acc), str(acc_dir))
            shutil.move(str(extracted_gyro), str(gyro_dir))
            logger.info(f"MotionSense data successfully downloaded to {ms_dir}")

        # ダウンロード完了を確認
        acc_files = list(acc_dir.glob('*/*.csv'))
        gyro_files = list(gyro_dir.glob('*/*.csv'))
        logger.info(f"Found {len(acc_files)} accelerometer CSV files")
        logger.info(f"Found {len(gyro_files)} gyroscope CSV files")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        全データファイルを読み込み、被験者単位にグループ化

        Returns:
            {subject_id: (data, labels)} の辞書
                data: (num_samples, 6) - ACC(3) + GYRO(3)
                labels: (num_samples,) - 活動ラベル
        """
        ms_dir = self.raw_data_path / 'motionsense'
        acc_dir = ms_dir / 'B_Accelerometer_data'
        gyro_dir = ms_dir / 'C_Gyroscope_data'

        if not acc_dir.exists() or not gyro_dir.exists():
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
                acc_trial_dir = acc_dir / f"{act_name}_{trial}"
                gyro_trial_dir = gyro_dir / f"{act_name}_{trial}"

                if not acc_trial_dir.exists() or not gyro_trial_dir.exists():
                    logger.warning(f"Trial directory not found: {act_name}_{trial}")
                    continue

                # 被験者ごとのCSVファイルを取得
                acc_files = sorted(acc_trial_dir.glob('sub_*.csv'))

                for acc_file in acc_files:
                    # ファイル名から被験者IDを取得: sub_1.csv -> 1
                    try:
                        subject_id = int(acc_file.stem.split('_')[1])
                    except (IndexError, ValueError):
                        logger.warning(f"Invalid filename format: {acc_file.name}")
                        continue

                    # 対応するジャイロファイル
                    gyro_file = gyro_trial_dir / acc_file.name

                    if not gyro_file.exists():
                        logger.warning(f"Gyroscope file not found: {gyro_file}")
                        continue

                    try:
                        # Accelerometer CSVを読み込み
                        df_acc = pd.read_csv(acc_file)
                        if 'Unnamed: 0' in df_acc.columns:
                            df_acc = df_acc.drop(['Unnamed: 0'], axis=1)
                        acc_data = df_acc[self.acc_columns].values  # (N, 3)

                        # Gyroscope CSVを読み込み
                        df_gyro = pd.read_csv(gyro_file)
                        if 'Unnamed: 0' in df_gyro.columns:
                            df_gyro = df_gyro.drop(['Unnamed: 0'], axis=1)
                        gyro_data = df_gyro[self.gyro_columns].values  # (N, 3)

                        # サンプル数を揃える（最小長に合わせる）
                        min_len = min(len(acc_data), len(gyro_data))
                        acc_data = acc_data[:min_len]
                        gyro_data = gyro_data[:min_len]

                        # 結合: (N, 6)
                        combined_data = np.hstack([acc_data, gyro_data])

                        # 全サンプルに同じラベルを付与
                        labels = np.full(len(combined_data), act_label, dtype=np.int32)

                        # 被験者ごとのリストに追加
                        subject_data_list[subject_id].append((combined_data, labels))
                        logger.info(f"Loaded {acc_file.name}: data={combined_data.shape}, label={act_label} ({act_name})")

                    except Exception as e:
                        logger.error(f"Error loading {acc_file}: {e}")

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

            # Pocketセンサー（6チャンネル: ACC(3) + GYRO(3)）
            sensor_name = 'Pocket'
            sensor_data = subject_data  # (N, 6)

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
            data/processed/motionsense/USER00001/Pocket/ACC/X.npy, Y.npy
            data/processed/motionsense/USER00001/Pocket/GYRO/X.npy, Y.npy
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
