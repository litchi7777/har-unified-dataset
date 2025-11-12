"""
USC-HAD (USC Human Activity Dataset) 前処理

USC-HAD データセット:
- 12種類の日常動作
- 14人の被験者（男性7人、女性7人）
- 1つのIMUセンサー（6チャンネル：3軸ACC + 3軸GYRO）
- サンプリングレート: 100Hz
- センサー位置: 前部右腰（Hip）
- 加速度単位: G（すでにG単位）
"""

import numpy as np
import scipy.io
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    get_class_distribution,
    resample_timeseries
)
from .common import (
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


@register_preprocessor('uschad')
class USCHADPreprocessor(BasePreprocessor):
    """
    USC-HADデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # USC-HAD固有の設定
        self.num_activities = 12
        self.num_subjects = 14
        self.num_sensors = 1  # Hip（前部右腰）のみ
        self.channels_per_sensor = 6  # 3-axis acc + 3-axis gyro
        self.num_channels = 6

        # サンプリングレート
        self.original_sampling_rate = 100  # Hz (USC-HADのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名とチャンネルマッピング
        self.sensor_names = ['Hip']
        self.sensor_channel_ranges = {
            'Hip': (0, 6),  # channels 0-5: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.modalities = ['ACC', 'GYRO']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3軸加速度 (G単位)
            'GYRO': (3, 6),  # 3軸ジャイロ (dps)
        }
        self.channels_per_modality = 3

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（ACC/GYRO）
        self.scale_factor = DATASETS.get('USCHAD', {}).get('scale_factor', None)
        self.gyro_scale_factor = DATASETS.get('USCHAD', {}).get('gyro_scale_factor', None)

        # アクティビティ名マッピング
        self.activity_names = {
            1: 'walking-forward',
            2: 'walking-left',
            3: 'walking-right',
            4: 'walking-upstairs',
            5: 'walking-downstairs',
            6: 'running-forward',
            7: 'jumping-up',
            8: 'sitting',
            9: 'standing',
            10: 'sleeping',
            11: 'elevator-up',
            12: 'elevator-down'
        }

    def get_dataset_name(self) -> str:
        return 'uschad'

    def download_dataset(self) -> None:
        """
        USC-HADデータセットのダウンロード

        USC-HADは手動ダウンロードが必要です。
        https://sipi.usc.edu/had/USC-HAD.zip からダウンロードして
        data/raw/uschad/ に配置してください。

        期待されるディレクトリ構造:
            data/raw/uschad/
            ├── Subject1/
            │   ├── a1t1.mat
            │   ├── a1t2.mat
            │   └── ...
            ├── Subject2/
            └── ...
        """
        uschad_raw_path = self.raw_data_path / self.dataset_name

        if check_dataset_exists(uschad_raw_path, required_files=['Subject*/a*.mat']):
            logger.info(f"USC-HAD data already exists at {uschad_raw_path}")
            return

        raise NotImplementedError(
            f"USC-HAD dataset must be manually downloaded.\n\n"
            f"Please follow these steps:\n"
            f"1. Download from: https://sipi.usc.edu/had/USC-HAD.zip\n"
            f"2. Extract the zip file\n"
            f"3. Move the 'USC-HAD' folder to: {uschad_raw_path}\n\n"
            f"Expected structure:\n"
            f"  {uschad_raw_path}/\n"
            f"  ├── Subject1/\n"
            f"  │   ├── a1t1.mat\n"
            f"  │   ├── a1t2.mat\n"
            f"  │   └── ...\n"
            f"  ├── Subject2/\n"
            f"  └── ..."
        )

    def _load_mat_file(self, mat_path: Path) -> Dict:
        """
        .mat ファイルを読み込む

        Args:
            mat_path: .mat ファイルのパス

        Returns:
            読み込んだデータの辞書
        """
        data = scipy.io.loadmat(mat_path)

        # メタデータを抽出（1要素の配列から値を取り出す）
        result = {
            "subject": data["subject"][0] if "subject" in data else "unknown",
            "activity": data["activity"][0] if "activity" in data else "unknown",
            "activity_number": int(data["activity_number"][0]) if "activity_number" in data else -1,
            "trial": int(data["trial"][0]) if "trial" in data else -1,
            "sensor_readings": data["sensor_readings"],  # shape: (N, 6)
        }

        return result

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        USC-HADの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/uschad/Subject1/a1t1.mat (activity 1, trial 1)
        - 各ファイル: sensor_readings (N, 6) - [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 6) の配列
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"USC-HAD raw data not found at {raw_path}\n"
                "Expected structure: data/raw/uschad/Subject1/a1t1.mat"
            )

        # 被験者ごとにデータを格納
        person_data = {}

        # 各被験者について
        for subject_id in range(1, self.num_subjects + 1):
            subject_dir = raw_path / f"Subject{subject_id}"

            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            person_data[subject_id] = {'data': [], 'labels': []}

            # 全ての .mat ファイルを読み込む
            mat_files = sorted(subject_dir.glob("*.mat"))

            if len(mat_files) == 0:
                logger.warning(f"No .mat files found in {subject_dir}")
                continue

            for mat_file in mat_files:
                try:
                    # データ読み込み
                    mat_data = self._load_mat_file(mat_file)
                    sensor_readings = mat_data["sensor_readings"]  # (N, 6)
                    activity_num = mat_data["activity_number"]

                    # データ形状チェック
                    if sensor_readings.shape[1] != self.num_channels:
                        logger.warning(
                            f"Unexpected number of channels in {mat_file}: "
                            f"{sensor_readings.shape[1]} (expected {self.num_channels})"
                        )
                        continue

                    # ラベル生成（0-indexed）
                    segment_labels = np.full(len(sensor_readings), activity_num - 1)

                    person_data[subject_id]['data'].append(sensor_readings)
                    person_data[subject_id]['labels'].append(segment_labels)

                except Exception as e:
                    logger.error(f"Error loading {mat_file}: {e}")
                    continue

        # 各被験者のデータを結合
        result = {}
        for subject_id in range(1, self.num_subjects + 1):
            if subject_id in person_data and person_data[subject_id]['data']:
                data = np.vstack(person_data[subject_id]['data'])
                labels = np.hstack(person_data[subject_id]['labels'])
                result[subject_id] = (data, labels)
                logger.info(f"USER{subject_id:05d}: {data.shape}, Labels: {labels.shape}")
            else:
                logger.warning(f"No data loaded for USER{subject_id:05d}")

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

            # リサンプリング (100Hz -> 30Hz)
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
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            例: {'Hip/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーについて処理（USC-HADはHipのみ）
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # センサーのチャンネルを抽出
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]  # (samples, 6)

                # スライディングウィンドウ適用（最後のウィンドウはパディング）
                # 正規化は行わず、生のセンサーデータを保持
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )
                # windowed_data: (num_windows, window_size, 6)

                # 各モダリティに分割
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, 3)

                    # スケーリング適用
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")
                    elif modality_name == 'GYRO' and self.gyro_scale_factor is not None:
                        modality_data = modality_data * self.gyro_scale_factor
                        logger.info(
                            f"  Applied gyro_scale_factor={self.gyro_scale_factor} to "
                            f"{sensor_name}/{modality_name}"
                        )

                    # 形状を変換: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
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
            data/processed/uschad/USER00001/Hip/ACC/X.npy, Y.npy
            data/processed/uschad/USER00001/Hip/GYRO/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'num_sensors': self.num_sensors,
            'sensor_names': self.sensor_names,
            'modalities': self.modalities,
            'channels_per_modality': self.channels_per_modality,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # 正規化なし（生データ保持）
            'scale_factor': self.scale_factor,  # USC-HADはすでにG単位なのでNone
            'gyro_scale_factor': self.gyro_scale_factor,  # ジャイロ変換係数
            'data_dtype': 'float16',  # データ型
            'data_shape': f'(num_windows, {self.channels_per_modality}, {self.window_size})',
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
                X = arrays['X']  # (num_windows, 3, window_size)
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
