"""
TMD (Transportation Mode Detection) 前処理

TMD データセット:
- 5種類の移動モード（Walking, Car, Still, Train, Bus）
- 16人の被験者（U1-U16）
- スマートフォンセンサー（加速度、ジャイロ等）
- イベントドリブンのサンプリング（可変レート）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from collections import defaultdict

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


@register_preprocessor('tmd')
class TMDPreprocessor(BasePreprocessor):
    """
    TMDデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # TMD固有の設定
        self.num_activities = 5
        self.num_subjects = 16

        # センサー名
        self.sensor_names = ['Smartphone']

        # モダリティ
        self.modalities = ['ACC', 'GYRO']
        self.channels_per_modality = {
            'ACC': 3,   # 3軸加速度
            'GYRO': 3   # 3軸ジャイロ
        }

        # アクティビティマッピング
        self.activity_map = {
            'Walking': 0,
            'Car': 1,
            'Still': 2,
            'Train': 3,
            'Bus': 4
        }

        # サンプリングレート（可変 -> 固定に変換）
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（m/s^2 -> G に変換）
        self.scale_factor = DATASETS.get('TMD', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'tmd'

    def download_dataset(self) -> None:
        """
        TMDデータセットは手動ダウンロードが必要
        """
        logger.info("=" * 80)
        logger.info("TMD dataset must be downloaded manually")
        logger.info("=" * 80)
        logger.info("Please download from: https://github.com/robieta/cs229_project")
        logger.info(f"Extract to: {self.raw_data_path / self.dataset_name / 'raw_data'}")
        raise NotImplementedError(
            "TMD dataset must be downloaded manually. "
            "Extract the 'raw_data' folder to data/raw/tmd/"
        )

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        TMDデータを被験者ごとに読み込む

        Returns:
            {subject_id: (data, labels)} の辞書
                data: (num_samples, 6) の配列（ACC 3ch + GYRO 3ch）
                labels: (num_samples,) の配列
        """
        logger.info("=" * 80)
        logger.info("Loading TMD raw data")
        logger.info("=" * 80)

        tmd_data_path = self.raw_data_path / self.dataset_name / 'raw_data'

        if not tmd_data_path.exists():
            raise FileNotFoundError(
                f"TMD data not found at {tmd_data_path}. "
                "Please download the dataset manually."
            )

        all_data = {}

        # 各被験者のデータを読み込み
        for subject_id in range(1, 17):  # U1 to U16
            subject_name = f"U{subject_id}"
            subject_path = tmd_data_path / subject_name

            if not subject_path.exists():
                logger.warning(f"Subject {subject_name} not found, skipping")
                continue

            logger.info(f"Loading {subject_name}")

            # 被験者のすべてのCSVファイルを読み込み
            subject_data_list = []
            subject_labels_list = []

            csv_files = list(subject_path.glob("sensorfile_*.csv"))

            for csv_file in csv_files:
                # ファイル名からアクティビティラベルを抽出
                # 例: sensorfile_U1_Walking_1480512323378.csv -> Walking
                filename_parts = csv_file.stem.split('_')
                if len(filename_parts) >= 3:
                    activity_name = filename_parts[2]
                    if activity_name in self.activity_map:
                        label = self.activity_map[activity_name]

                        # CSVファイルを読み込んでセンサーデータを抽出
                        try:
                            data = self._parse_csv_file(csv_file)
                            if data is not None and len(data) > 0:
                                labels = np.full(len(data), label, dtype=np.int32)
                                subject_data_list.append(data)
                                subject_labels_list.append(labels)
                        except Exception as e:
                            logger.warning(f"Failed to parse {csv_file.name}: {e}")
                            continue

            if subject_data_list:
                # 被験者のすべてのデータを結合
                subject_data = np.vstack(subject_data_list)
                subject_labels = np.concatenate(subject_labels_list)

                all_data[subject_id] = (subject_data, subject_labels)
                logger.info(
                    f"  {subject_name}: {subject_data.shape[0]} samples, "
                    f"{len(np.unique(subject_labels))} activities"
                )

        logger.info(f"Loaded {len(all_data)} subjects")
        return all_data

    def _parse_csv_file(self, csv_file: Path) -> np.ndarray:
        """
        TMDのCSVファイルをパースしてセンサーデータを抽出

        Args:
            csv_file: CSVファイルのパス

        Returns:
            (num_samples, 6) の配列（ACC 3ch + GYRO 3ch）
        """
        # TMDのCSVは各行で異なるカラム数を持つため、手動でパース
        return self._parse_csv_manual(csv_file)

    def _parse_csv_manual(self, csv_file: Path) -> np.ndarray:
        """
        手動でCSVをパース（pandas でエラーが出る場合）
        """
        acc_data = []
        gyro_data = []

        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    try:
                        timestamp = float(parts[0])
                        sensor_type = parts[1]
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4]) if len(parts) > 4 else 0.0

                        if sensor_type == 'android.sensor.accelerometer':
                            acc_data.append([timestamp, x, y, z])
                        elif sensor_type == 'android.sensor.gyroscope':
                            gyro_data.append([timestamp, x, y, z])
                    except ValueError:
                        continue

        if not acc_data or not gyro_data:
            return None

        acc_df = pd.DataFrame(acc_data, columns=['timestamp', 'x', 'y', 'z'])
        gyro_df = pd.DataFrame(gyro_data, columns=['timestamp', 'x', 'y', 'z'])

        # タイムスタンプでソートして重複を除去
        # スプライン補間には厳密に単調増加する系列が必要
        acc_df = acc_df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)
        gyro_df = gyro_df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first').reset_index(drop=True)

        return self._align_sensor_data(acc_df, gyro_df)

    def _align_sensor_data(self, acc_df: pd.DataFrame, gyro_df: pd.DataFrame) -> np.ndarray:
        """
        加速度とジャイロのタイムスタンプを揃えてデータを統合

        論文記載: TMDは20Hzで測定されたデータ（実際はイベントドリブンで疎）
        高密度スプライン補間（1ms）+ ローパスフィルタ + ダウンサンプリングで滑らかな30Hz信号を生成

        Args:
            acc_df: 加速度データ（timestamp, x, y, z）
            gyro_df: ジャイロデータ（timestamp, x, y, z）

        Returns:
            (num_samples, 6) の配列（ACC 3ch + GYRO 3ch）
        """
        from scipy import signal, interpolate

        # 共通のタイムスタンプ範囲を取得
        min_time = max(acc_df['timestamp'].min(), gyro_df['timestamp'].min())
        max_time = min(acc_df['timestamp'].max(), gyro_df['timestamp'].max())

        duration = (max_time - min_time) / 1000.0  # 秒

        if duration <= 0:
            return None

        # ステップ1: 高密度スプライン補間（1ms = 1000Hz）
        # スプライン補間は高密度でやった方が滑らか
        high_rate = 1000  # Hz (1ms間隔)
        num_high_samples = int(duration * high_rate)

        if num_high_samples <= 0:
            return None

        high_res_times = np.linspace(min_time, max_time, num_high_samples)

        # 3次スプライン補間で高密度化（線形補間より滑らか）
        acc_high = np.zeros((num_high_samples, 3))
        gyro_high = np.zeros((num_high_samples, 3))

        for i, col in enumerate(['x', 'y', 'z']):
            # 加速度計: 3次スプライン補間（k=3で滑らか）
            # データ点が少ない場合は次数を下げる
            k_acc = min(3, len(acc_df) - 1)
            if k_acc >= 1:
                spline_acc = interpolate.make_interp_spline(
                    acc_df['timestamp'].values,
                    acc_df[col].values,
                    k=k_acc
                )
                acc_high[:, i] = spline_acc(high_res_times)
            else:
                # データ点が1つ以下の場合は定数値で埋める
                acc_high[:, i] = acc_df[col].values[0] if len(acc_df) > 0 else 0.0

            # ジャイロ: 3次スプライン補間
            k_gyro = min(3, len(gyro_df) - 1)
            if k_gyro >= 1:
                spline_gyro = interpolate.make_interp_spline(
                    gyro_df['timestamp'].values,
                    gyro_df[col].values,
                    k=k_gyro
                )
                gyro_high[:, i] = spline_gyro(high_res_times)
            else:
                gyro_high[:, i] = gyro_df[col].values[0] if len(gyro_df) > 0 else 0.0

        # ステップ2: ローパスフィルタを適用（高周波ノイズ除去、さらに滑らかに）
        # カットオフ周波数: 10Hz（論文記載の20Hzより低く設定）
        nyquist = high_rate / 2.0
        cutoff = 10.0  # Hz
        normalized_cutoff = cutoff / nyquist

        # バターワースフィルタ（4次）
        b, a = signal.butter(4, normalized_cutoff, btype='low')

        for i in range(3):
            acc_high[:, i] = signal.filtfilt(b, a, acc_high[:, i])
            gyro_high[:, i] = signal.filtfilt(b, a, gyro_high[:, i])

        # ステップ3: 1000Hz → 30Hz にダウンサンプリング（ポリフェーズフィルタ）
        target_rate = self.target_sampling_rate  # 30 Hz

        from math import gcd
        up = target_rate
        down = high_rate
        common_divisor = gcd(up, down)
        up = up // common_divisor
        down = down // common_divisor

        # 各チャンネルをポリフェーズフィルタリング
        acc_resampled = np.zeros((signal.resample_poly(acc_high[:, 0], up, down).shape[0], 3))
        gyro_resampled = np.zeros((signal.resample_poly(gyro_high[:, 0], up, down).shape[0], 3))

        for i in range(3):
            acc_resampled[:, i] = signal.resample_poly(acc_high[:, i], up, down)
            gyro_resampled[:, i] = signal.resample_poly(gyro_high[:, i], up, down)

        # ACC + GYROを結合
        combined = np.hstack([acc_resampled, gyro_resampled])

        return combined

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニング

        注: TMDはすでに_align_sensor_data()で30Hzにリサンプリング済みのため、
        ここでは追加のリサンプリングは不要
        """
        logger.info("=" * 80)
        logger.info("Cleaning data (already resampled to 30Hz)")
        logger.info("=" * 80)

        cleaned = {}

        for subject_id, (subject_data, labels) in data.items():
            # 無効なサンプルを除去（NaN、Inf）
            valid_mask = ~(np.isnan(subject_data).any(axis=1) | np.isinf(subject_data).any(axis=1))
            cleaned_data = subject_data[valid_mask]
            cleaned_labels = labels[valid_mask]

            if len(cleaned_data) == 0:
                logger.warning(f"Subject {subject_id}: No valid data after cleaning")
                continue

            cleaned[subject_id] = (cleaned_data, cleaned_labels)
            logger.info(
                f"Subject {subject_id:02d}: {subject_data.shape[0]} -> {len(cleaned_data)} samples "
                f"(cleaned, already at {self.target_sampling_rate}Hz)"
            )

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Returns:
            {subject_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        logger.info("=" * 80)
        logger.info("Extracting features (windowing and scaling)")
        logger.info("=" * 80)

        processed = {}

        for subject_id, (subject_data, labels) in data.items():
            logger.info(f"Processing Subject {subject_id:02d}")
            processed[subject_id] = {}

            # スライディングウィンドウ適用
            windowed_data, windowed_labels = create_sliding_windows(
                subject_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            # ACC（0-2）とGYRO（3-5）に分割
            for modality_name, ch_range in [('ACC', (0, 3)), ('GYRO', (3, 6))]:
                modality_data = windowed_data[:, :, ch_range[0]:ch_range[1]]

                # スケーリング適用（ACCのみ、scale_factorが定義されている場合）
                if modality_name == 'ACC' and self.scale_factor is not None:
                    modality_data = modality_data / self.scale_factor
                    logger.info(f"  Applied scale_factor={self.scale_factor} to Smartphone/{modality_name}")

                # 形状変換: (N, T, C) -> (N, C, T)
                modality_data = np.transpose(modality_data, (0, 2, 1))

                # float16に変換（メモリ効率化）
                modality_data = modality_data.astype(np.float16)

                sensor_modality_key = f"Smartphone/{modality_name}"
                processed[subject_id][sensor_modality_key] = {
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

        保存形式:
            data/processed/tmd/USER00001/Smartphone/ACC/X.npy, Y.npy
        """
        logger.info("=" * 80)
        logger.info("Saving processed data")
        logger.info("=" * 80)

        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_subjects': self.num_subjects,
            'num_sensors': 1,
            'sensor_names': self.sensor_names,
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
        logger.info("=" * 80)
        logger.info(f"SUCCESS: Preprocessing completed -> {base_path}")
        logger.info("=" * 80)
