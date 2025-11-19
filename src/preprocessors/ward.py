"""
WARD (Wearable Action Recognition Database) 前処理

WARD データセット:
- 13種類の日常動作
- 20人の被験者
- 5つのセンサー（左腕、右腕、腰、左足首、右足首）
- サンプリングレート: 20Hz
- 各センサー: 3軸加速度 + 2軸ジャイロスコープ
"""

import numpy as np
import scipy.io as sio
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
    download_file,
    extract_archive,
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# WARD データセットのURL
WARD_URL = "https://people.eecs.berkeley.edu/~yang/software/WAR/WARD1.zip"


@register_preprocessor('ward')
class WARDPreprocessor(BasePreprocessor):
    """
    WARDデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # WARD固有の設定
        self.num_activities = 13
        self.num_subjects = 20
        self.num_sensors = 5

        # サンプリングレート
        self.original_sampling_rate = 20  # Hz (WARDのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名とインデックスマッピング
        # PDF p.3 より: Sensor1=LeftArm, Sensor2=RightArm, Sensor3=Waist, Sensor4=LeftAnkle, Sensor5=RightAnkle
        self.sensor_names = ['LeftArm', 'RightArm', 'Waist', 'LeftAnkle', 'RightAnkle']
        self.sensor_indices = {
            'LeftArm': 0,      # Sensor 1
            'RightArm': 1,     # Sensor 2
            'Waist': 2,        # Sensor 3
            'LeftAnkle': 3,    # Sensor 4
            'RightAnkle': 4    # Sensor 5
        }

        # モダリティ（各センサーのチャンネル構成）
        # 各センサー: [ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y] の5チャンネル
        self.modalities = ['ACC', 'GYRO']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3軸加速度
            'GYRO': (3, 5),  # 2軸ジャイロ
        }

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz (リサンプリング後)
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（12-bit digital values (±2g) -> G に変換）
        # 12-bit signed: -2048 to 2047 for ±2g → 1024 = 1g
        self.scale_factor = DATASETS.get('WARD', {}).get('scale_factor', None)

        # 活動名マッピング（論文のTable 1より）
        self.activity_names = [
            'Stand',              # a1
            'Sit',                # a2
            'Lie',                # a3
            'Walk Forward',       # a4
            'Walk Left-Circle',   # a5
            'Walk Right-Circle',  # a6
            'Turn Left',          # a7
            'Turn Right',         # a8
            'Go Upstairs',        # a9
            'Go Downstairs',      # a10
            'Jog',                # a11
            'Jump',               # a12
            'Push Wheelchair'     # a13
        ]

    def get_dataset_name(self) -> str:
        return 'ward'

    def download_dataset(self) -> None:
        """
        WARDデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading WARD dataset")
        logger.info("=" * 80)

        ward_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(ward_raw_path, required_files=['WARD1.0']):
            logger.warning(f"WARD data already exists at {ward_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive")
            ward_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = ward_raw_path.parent / 'WARD1.zip'
            download_file(WARD_URL, zip_path, desc='Downloading WARD')

            # 2. 解凍
            logger.info("Step 2/2: Extracting archive")
            extract_archive(zip_path, ward_raw_path.parent, desc='Extracting WARD')

            # WARD1.0 を ward にリネーム
            extracted_dir = ward_raw_path.parent / 'WARD1.0'
            if extracted_dir.exists() and not ward_raw_path.exists():
                extracted_dir.rename(ward_raw_path)

            # クリーンアップ
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: WARD dataset downloaded to {ward_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download WARD dataset: {e}", exc_info=True)
            raise

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        WARDの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/ward/Subject1/a1t1.mat (activity 1, trial 1)
        - MATファイル内: WearableData構造体
          - Class: 活動クラス (1-13)
          - Subject: 被験者番号 (1-20)
          - Reading: 5つのセンサーデータ (各センサー: (samples, 5))

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 5, 5) の配列 [5 sensors × 5 channels]
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"WARD raw data not found at {raw_path}\n"
                "Expected structure: data/raw/ward/Subject1/a1t1.mat"
            )

        # 被験者ごとにデータを格納
        person_data = {person_id: {'segments': []}
                       for person_id in range(1, self.num_subjects + 1)}

        # 各被験者のディレクトリを処理
        for person_id in range(1, self.num_subjects + 1):
            subject_dir = raw_path / f"Subject{person_id}"

            if not subject_dir.exists():
                logger.warning(f"Subject directory not found: {subject_dir}")
                continue

            # 全MATファイルを読み込み
            mat_files = sorted(subject_dir.glob("*.mat"))

            for mat_file in mat_files:
                try:
                    # MATファイル読み込み
                    mat = sio.loadmat(mat_file)
                    wd = mat['WearableData']

                    # クラスID取得 (1-13 -> 0-12)
                    class_id = int(wd['Class'][0, 0][0]) - 1

                    # Reading (sensor data) 取得
                    reading = wd['Reading'][0, 0]

                    # 5つのセンサーデータを結合
                    # 各センサー: (samples, 5) -> 全センサー: (5, samples, 5)
                    sensor_arrays = []
                    sample_lengths = []

                    for sensor_idx in range(5):
                        sensor_data = reading[0][sensor_idx]  # (samples, 5)

                        # Infチェック（欠損データ）
                        if np.isinf(sensor_data).any():
                            logger.warning(
                                f"Inf values detected in {mat_file.name}, "
                                f"Sensor {sensor_idx+1} - skipping this trial"
                            )
                            break

                        sensor_arrays.append(sensor_data)
                        sample_lengths.append(sensor_data.shape[0])
                    else:
                        # すべてのセンサーデータが有効
                        # サンプル数が揃っているか確認
                        if len(set(sample_lengths)) != 1:
                            logger.warning(
                                f"Sample length mismatch in {mat_file.name}: {sample_lengths} - skipping"
                            )
                            continue

                        num_samples = sample_lengths[0]

                        # センサーデータを統合: (samples, 5_sensors, 5_channels)
                        # sensor_arrays: list of 5 × (samples, 5)
                        combined_data = np.stack(sensor_arrays, axis=1)  # (samples, 5_sensors, 5_channels)

                        # ラベル生成
                        labels = np.full(num_samples, class_id, dtype=np.int32)

                        person_data[person_id]['segments'].append({
                            'data': combined_data,
                            'labels': labels,
                            'file': mat_file.name
                        })

                except Exception as e:
                    logger.error(f"Error loading {mat_file}: {e}")
                    continue

        # 各被験者のセグメントを結合
        result = {}
        for person_id in range(1, self.num_subjects + 1):
            segments = person_data[person_id]['segments']
            if segments:
                # 全セグメントを時間軸で結合
                all_data = np.concatenate([seg['data'] for seg in segments], axis=0)
                all_labels = np.concatenate([seg['labels'] for seg in segments], axis=0)

                result[person_id] = (all_data, all_labels)
                logger.info(
                    f"USER{person_id:05d}: {all_data.shape}, "
                    f"Labels: {all_labels.shape}, Segments: {len(segments)}"
                )
            else:
                logger.warning(f"No valid data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニングとリサンプリング

        Args:
            data: {person_id: (data, labels)} の辞書
                  data: (samples, 5_sensors, 5_channels)

        Returns:
            クリーニング・リサンプリング済み {person_id: (data, labels)}
        """
        cleaned = {}
        for person_id, (person_data, labels) in data.items():
            # 形状を一時的に変換: (samples, 5, 5) -> (samples, 25)
            original_shape = person_data.shape
            flat_data = person_data.reshape(original_shape[0], -1)

            # 無効なサンプルを除去
            cleaned_data, cleaned_labels = filter_invalid_samples(flat_data, labels)

            # リサンプリング (20Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                # 形状を復元: (samples, 25) -> (samples, 5, 5)
                resampled_data = resampled_data.reshape(-1, 5, 5)
                cleaned[person_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{person_id:05d} cleaned and resampled: {resampled_data.shape}")
            else:
                # 形状を復元
                cleaned_data = cleaned_data.reshape(-1, 5, 5)
                cleaned[person_id] = (cleaned_data, cleaned_labels)
                logger.info(f"USER{person_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Args:
            data: {person_id: (data, labels)} の辞書
                  data: (samples, 5_sensors, 5_channels)

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            例: {'LeftArm/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_name in self.sensor_names:
                sensor_idx = self.sensor_indices[sensor_name]

                # センサーデータを抽出: (samples, 5_channels)
                sensor_data = person_data[:, sensor_idx, :]  # (samples, 5)

                # スライディングウィンドウ適用
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )
                # windowed_data: (num_windows, window_size, 5)

                # 各モダリティに分割
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # スケーリング適用（加速度のみ）
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # 形状を変換: (num_windows, window_size, channels) -> (num_windows, channels, window_size)
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
            data/processed/ward/USER00001/LeftArm/ACC/X.npy, Y.npy
            data/processed/ward/USER00001/LeftArm/GYRO/X.npy, Y.npy
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
            'modalities': self.modalities,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # 正規化なし（生データ保持）
            'scale_factor': self.scale_factor,  # スケーリング係数（ACCのみ適用）
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
                X = arrays['X']  # (num_windows, channels, window_size)
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
