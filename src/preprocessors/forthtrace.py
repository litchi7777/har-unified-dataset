"""
FORTHTRACE (FORTH-TRACE) データセット前処理

FORTHTRACE データセット:
- 16種類の活動（基本7 + 姿勢遷移9）
- 15人の被験者
- 5つのShimmerセンサー（9チャンネル×5）
- サンプリングレート: 51.2Hz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import shutil

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


# FORTHTRACE データセットのURL
FORTHTRACE_URL = "https://zenodo.org/records/841301/files/FORTH_TRACE_DATASET.zip?download=1"


@register_preprocessor('forthtrace')
class ForthtracePreprocessor(BasePreprocessor):
    """
    FORTHTRACEデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # FORTHTRACE固有の設定
        self.num_activities = 16
        self.num_subjects = 15
        self.num_sensors = 5
        self.channels_per_sensor = 9  # 3-axis acc, gyro, mag
        self.num_channels = 9  # 各CSVファイルは1センサー分

        # サンプリングレート
        self.original_sampling_rate = 51.2  # Hz (FORTHTRACEのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名（デバイスID1-5に対応）
        self.sensor_names = ['LeftWrist', 'RightWrist', 'Torso', 'RightThigh', 'LeftAnkle']
        self.device_id_to_sensor = {
            1: 'LeftWrist',
            2: 'RightWrist',
            3: 'Torso',
            4: 'RightThigh',
            5: 'LeftAnkle'
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.modalities = ['ACC', 'GYRO', 'MAG']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3軸加速度 (columns 1,2,3)
            'GYRO': (3, 6),  # 3軸ジャイロ (columns 4,5,6)
            'MAG': (6, 9)    # 3軸地磁気 (columns 7,8,9)
        }
        self.channels_per_modality = 3

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数
        self.scale_factor = DATASETS.get('FORTHTRACE', {}).get('scale_factor', None)  # ACC: m/s^2 -> G
        self.gyro_scale_factor = DATASETS.get('FORTHTRACE', {}).get('gyro_scale_factor', None)  # GYRO: deg/s -> rad/s

    def get_dataset_name(self) -> str:
        return 'forthtrace'

    def download_dataset(self) -> None:
        """
        FORTHTRACEデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading FORTHTRACE dataset")
        logger.info("=" * 80)

        forthtrace_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(forthtrace_raw_path, required_files=['part*/part*dev*.csv']):
            logger.warning(f"FORTHTRACE data already exists at {forthtrace_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive")
            forthtrace_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = forthtrace_raw_path.parent / 'forthtrace.zip'
            download_file(FORTHTRACE_URL, zip_path, desc='Downloading FORTHTRACE')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = forthtrace_raw_path.parent / 'forthtrace_temp'
            extract_archive(zip_path, extract_to, desc='Extracting FORTHTRACE')
            self._organize_forthtrace_data(extract_to, forthtrace_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: FORTHTRACE dataset downloaded to {forthtrace_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download FORTHTRACE dataset: {e}", exc_info=True)
            raise

    def _organize_forthtrace_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        FORTHTRACEデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/forthtrace）
        """
        logger.info(f"Organizing FORTHTRACE data from {extracted_path} to {target_path}")

        # 解凍されたデータのルートを見つける
        data_root = extracted_path

        # "FORTH_TRACE_DATASET-master" フォルダを探す
        if (extracted_path / "FORTH_TRACE_DATASET-master").exists():
            data_root = extracted_path / "FORTH_TRACE_DATASET-master"
        elif (extracted_path / "FORTH_TRACE_DATASET").exists():
            data_root = extracted_path / "FORTH_TRACE_DATASET"

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find data directory in {extracted_path}")

        # part0, part1, ... などのディレクトリを探してコピー
        part_dirs = sorted([d for d in data_root.glob("part*") if d.is_dir()])

        if not part_dirs:
            raise FileNotFoundError(f"Could not find part directories in {data_root}")

        from tqdm import tqdm
        for part_dir in tqdm(part_dirs, desc='Organizing participants'):
            part_name = part_dir.name
            target_part_dir = target_path / part_name

            # パートディレクトリをコピー
            if target_part_dir.exists():
                shutil.rmtree(target_part_dir)
            shutil.copytree(part_dir, target_part_dir)

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        FORTHTRACEの生データを被験者ごとにセンサー別に読み込む

        想定フォーマット:
        - data/raw/forthtrace/part0/part0dev1.csv (participant 0, device 1)
        - 各CSVファイル: 12列（デバイスID, acc_x/y/z, gyro_x/y/z, mag_x/y/z, timestamp, label）

        Returns:
            person_data: {person_id: {sensor_name: (data, labels)}} の辞書
                data: (num_samples, 9) の配列（acc, gyro, mag）
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"FORTHTRACE raw data not found at {raw_path}\n"
                "Expected structure: data/raw/forthtrace/part0/part0dev1.csv"
            )

        # 被験者ごとにデータを格納
        person_data = {}

        # 各被験者について (part0 ~ part14 = 15 subjects)
        # person_idは1-indexedで管理（USER00001から開始）
        for part_idx in range(self.num_subjects):
            person_dir = raw_path / f"part{part_idx}"

            if not person_dir.exists():
                logger.warning(f"Participant directory not found: {person_dir}")
                continue

            # person_idを1-indexedに変換（part0 -> USER00001）
            person_id = part_idx + 1
            person_data[person_id] = {}

            # 各デバイスについて（device 1-5）
            for device_id in range(1, self.num_sensors + 1):
                device_file = person_dir / f"part{part_idx}dev{device_id}.csv"

                if not device_file.exists():
                    logger.warning(f"Device file not found: {device_file}")
                    continue

                try:
                    # CSVデータ読み込み（ヘッダーなし）
                    df = pd.read_csv(device_file, header=None)

                    # 列の検証
                    if df.shape[1] != 12:
                        logger.warning(
                            f"Unexpected number of columns in {device_file}: "
                            f"{df.shape[1]} (expected 12)"
                        )
                        continue

                    # センサーデータ抽出（列1-9: acc, gyro, mag）
                    sensor_data = df.iloc[:, 1:10].values.astype(np.float32)

                    # ラベル抽出（列11、1-indexedなので0-indexedに変換）
                    labels = df.iloc[:, 11].values.astype(int) - 1

                    # センサー名取得
                    sensor_name = self.device_id_to_sensor[device_id]

                    person_data[person_id][sensor_name] = (sensor_data, labels)

                    logger.info(
                        f"USER{person_id:05d}/{sensor_name}: {sensor_data.shape}, "
                        f"Labels: {labels.shape}"
                    )

                except Exception as e:
                    logger.error(f"Error loading {device_file}: {e}")
                    continue

        if not person_data:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(person_data)}")
        return person_data

    def clean_data(
        self, data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        データのクリーニングとリサンプリング

        Args:
            data: {person_id: {sensor_name: (data, labels)}} の辞書

        Returns:
            クリーニング・リサンプリング済み {person_id: {sensor_name: (data, labels)}}
        """
        cleaned = {}
        for person_id, sensors in data.items():
            cleaned[person_id] = {}

            for sensor_name, (sensor_data, labels) in sensors.items():
                # 無効なサンプルを除去
                cleaned_data, cleaned_labels = filter_invalid_samples(sensor_data, labels)

                # リサンプリング (51.2Hz -> 30Hz)
                if self.original_sampling_rate != self.target_sampling_rate:
                    resampled_data, resampled_labels = resample_timeseries(
                        cleaned_data,
                        cleaned_labels,
                        self.original_sampling_rate,
                        self.target_sampling_rate
                    )
                    cleaned[person_id][sensor_name] = (resampled_data, resampled_labels)
                    logger.info(
                        f"USER{person_id:05d}/{sensor_name} cleaned and resampled: "
                        f"{resampled_data.shape}"
                    )
                else:
                    cleaned[person_id][sensor_name] = (cleaned_data, cleaned_labels)
                    logger.info(f"USER{person_id:05d}/{sensor_name} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(
        self, data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Args:
            data: {person_id: {sensor_name: (data, labels)}} の辞書

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            例: {'LeftWrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, sensors in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_name, (sensor_data, labels) in sensors.items():
                # スライディングウィンドウ適用
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )
                # windowed_data: (num_windows, window_size, 9)

                # 各モダリティに分割
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, 3)

                    # スケーリング適用（加速度のみ）
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(
                            f"  Applied acc_scale_factor={self.scale_factor} to "
                            f"{sensor_name}/{modality_name}"
                        )
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

    def save_processed_data(
        self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        処理済みデータを保存

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        保存形式:
            data/processed/forthtrace/USER00000/LeftWrist/ACC/X.npy, Y.npy
            data/processed/forthtrace/USER00000/LeftWrist/GYRO/X.npy, Y.npy
            data/processed/forthtrace/USER00000/Torso/ACC/X.npy, Y.npy
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
            'channels_per_modality': self.channels_per_modality,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # 正規化なし（生データ保持）
            'scale_factor': self.scale_factor,  # スケーリング係数（ACCのみ適用）
            'gyro_scale_factor': self.gyro_scale_factor,  # ジャイロ変換
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
