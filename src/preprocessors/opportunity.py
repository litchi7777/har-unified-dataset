"""
OPPORTUNITY (OPPORTUNITY Activity Recognition Dataset) 前処理

OPPORTUNITY データセット:
- 17種類のmid-level gestures (+ Null class)
- 4人の被験者
- 5つのIMUセンサー（Body-worn）
- サンプリングレート: 30Hz
"""

import numpy as np
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


# OPPORTUNITY データセットのURL
OPPORTUNITY_URL = "https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip"


# センサー列のマッピング（0-indexed）
# Body-worn IMU sensors: BACK, RUA (Right Upper Arm), RLA (Right Lower Arm),
#                        LUA (Left Upper Arm), LLA (Left Lower Arm)
# 各IMUは17チャンネル: ACC(3), GYRO(3), MAG(3), QUAT(4), other(4)
SENSOR_COLUMNS = {
    'BACK': {
        'start': 1,   # column 1 in file (0-indexed in numpy after loading)
        'ACC': (1, 4),     # columns 1-3 (ACC X, Y, Z)
        'GYRO': (4, 7),    # columns 4-6 (GYRO X, Y, Z)
        'MAG': (7, 10),    # columns 7-9 (MAG X, Y, Z)
    },
    'RUA': {
        'start': 18,
        'ACC': (18, 21),
        'GYRO': (21, 24),
        'MAG': (24, 27),
    },
    'RLA': {
        'start': 35,
        'ACC': (35, 38),
        'GYRO': (38, 41),
        'MAG': (41, 44),
    },
    'LUA': {
        'start': 52,
        'ACC': (52, 55),
        'GYRO': (55, 58),
        'MAG': (58, 61),
    },
    'LLA': {
        'start': 69,
        'ACC': (69, 72),
        'GYRO': (72, 75),
        'MAG': (75, 78),
    },
}

# ラベル列（mid-level gestures）
LABEL_COLUMN = 244  # column 244 (0-indexed: 243)


@register_preprocessor('opportunity')
class OpportunityPreprocessor(BasePreprocessor):
    """
    OPPORTUNITYデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # OPPORTUNITY固有の設定
        self.num_activities = 17  # mid-level gestures
        self.num_subjects = 4
        self.num_sensors = 5  # BACK, RUA, RLA, LUA, LLA

        # サンプリングレート
        self.original_sampling_rate = 30  # Hz (OPPORTUNITYのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名とチャンネルマッピング
        self.sensor_names = ['BACK', 'RUA', 'RLA', 'LUA', 'LLA']
        self.sensor_columns = SENSOR_COLUMNS

        # モダリティ
        self.modalities = ['ACC', 'GYRO', 'MAG']
        self.channels_per_modality = 3

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（m/s^2 -> G に変換）
        self.scale_factor = DATASETS.get('OPPORTUNITY', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'opportunity'

    def download_dataset(self) -> None:
        """
        OPPORTUNITYデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading OPPORTUNITY dataset")
        logger.info("=" * 80)

        opportunity_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(opportunity_raw_path, required_files=['*.dat']):
            logger.warning(f"OPPORTUNITY data already exists at {opportunity_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive")
            opportunity_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = opportunity_raw_path.parent / 'opportunity.zip'
            download_file(OPPORTUNITY_URL, zip_path, desc='Downloading OPPORTUNITY')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = opportunity_raw_path.parent / 'opportunity_temp'
            extract_archive(zip_path, extract_to, desc='Extracting OPPORTUNITY')
            self._organize_opportunity_data(extract_to, opportunity_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: OPPORTUNITY dataset downloaded to {opportunity_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download OPPORTUNITY dataset: {e}", exc_info=True)
            raise

    def _organize_opportunity_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        OPPORTUNITYデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/opportunity）
        """
        import shutil
        from tqdm import tqdm

        logger.info(f"Organizing OPPORTUNITY data from {extracted_path} to {target_path}")

        # 解凍されたデータのルートを見つける
        data_root = extracted_path

        # "OpportunityUCIDataset" フォルダを探す
        if (extracted_path / "OpportunityUCIDataset").exists():
            data_root = extracted_path / "OpportunityUCIDataset"
            if (data_root / "dataset").exists():
                data_root = data_root / "dataset"

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find data directory in {extracted_path}")

        # .datファイルを探してコピー
        dat_files = list(data_root.glob("*.dat"))

        if not dat_files:
            # サブディレクトリも探す
            dat_files = list(data_root.glob("**/*.dat"))

        if not dat_files:
            raise FileNotFoundError(f"No .dat files found in {data_root}")

        logger.info(f"Found {len(dat_files)} .dat files")

        for dat_file in tqdm(dat_files, desc='Organizing files'):
            target_file = target_path / dat_file.name
            if target_file.exists():
                target_file.unlink()
            shutil.copy2(dat_file, target_file)

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        OPPORTUNITYの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/opportunity/S1-ADL1.dat, S1-ADL2.dat, ..., S1-Drill.dat
        - 各ファイル: (samples, 250) のテキストファイル（スペース区切り）

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, num_channels) の配列（選択されたセンサー列のみ）
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"OPPORTUNITY raw data not found at {raw_path}\n"
                "Expected structure: data/raw/opportunity/S1-ADL1.dat"
            )

        # 被験者ごとにデータを格納
        person_data = {person_id: {'data': [], 'labels': []}
                       for person_id in range(1, self.num_subjects + 1)}

        # 各被験者のファイルを読み込む
        for person_id in range(1, self.num_subjects + 1):
            # ADLファイルとDrillファイルを読み込む
            file_patterns = [
                f"S{person_id}-ADL*.dat",
                f"S{person_id}-Drill.dat"
            ]

            subject_files = []
            for pattern in file_patterns:
                subject_files.extend(sorted(raw_path.glob(pattern)))

            if not subject_files:
                logger.warning(f"No data files found for subject S{person_id}")
                continue

            logger.info(f"Loading {len(subject_files)} files for USER{person_id:05d}")

            for data_file in subject_files:
                try:
                    # データ読み込み（スペース区切り、NaNをnanに変換）
                    data = np.loadtxt(data_file, dtype=np.float32)

                    if data.ndim == 1:
                        data = data.reshape(1, -1)

                    logger.info(f"  Loaded {data_file.name}: {data.shape}")

                    # ラベル抽出（mid-level gestures: column 244, 0-indexed: 243）
                    if data.shape[1] <= LABEL_COLUMN - 1:
                        logger.warning(f"  File {data_file.name} has insufficient columns: {data.shape[1]}")
                        continue

                    labels = data[:, LABEL_COLUMN - 1].astype(np.int32)

                    # ラベル変換: 0 -> -1 (Null class), その他は1から0-indexedに変換
                    labels = np.where(labels == 0, -1, labels - 1)

                    # 必要なセンサー列のみを抽出して結合
                    sensor_data_list = []
                    for sensor_name in self.sensor_names:
                        for modality in self.modalities:
                            col_start, col_end = self.sensor_columns[sensor_name][modality]
                            # 0-indexed (列番号 - 1)
                            sensor_modality_data = data[:, col_start - 1:col_end - 1]
                            sensor_data_list.append(sensor_modality_data)

                    # 全センサー・モダリティを結合
                    combined_data = np.hstack(sensor_data_list)  # (samples, 45) - 5 sensors × 3 modalities × 3 channels

                    person_data[person_id]['data'].append(combined_data)
                    person_data[person_id]['labels'].append(labels)

                except Exception as e:
                    logger.error(f"Error loading {data_file}: {e}")
                    continue

        # 各被験者のデータを結合
        result = {}
        for person_id in range(1, self.num_subjects + 1):
            if person_data[person_id]['data']:
                data = np.vstack(person_data[person_id]['data'])
                labels = np.hstack(person_data[person_id]['labels'])
                result[person_id] = (data, labels)
                logger.info(f"USER{person_id:05d}: {data.shape}, Labels: {labels.shape}, Unique labels: {np.unique(labels)}")
            else:
                logger.warning(f"No data loaded for USER{person_id:05d}")

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニング

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            クリーニング済み {person_id: (data, labels)}
        """
        cleaned = {}
        for person_id, (person_data, labels) in data.items():
            # NaN/Infを含む行を除去
            cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)

            # サンプリングレートは既に30Hzなのでリサンプリング不要
            cleaned[person_id] = (cleaned_data, cleaned_labels)
            logger.info(f"USER{person_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # データは既に (samples, 45) の形式
            # 5 sensors × 3 modalities × 3 channels = 45 channels
            channel_idx = 0

            for sensor_name in self.sensor_names:
                for modality_name in self.modalities:
                    # 各センサー・モダリティの3チャンネルを抽出
                    sensor_modality_data = person_data[:, channel_idx:channel_idx + 3]
                    channel_idx += 3

                    # スライディングウィンドウ適用
                    windowed_data, windowed_labels = create_sliding_windows(
                        sensor_modality_data,
                        labels,
                        window_size=self.window_size,
                        stride=self.stride,
                        drop_last=False,
                        pad_last=True
                    )
                    # windowed_data: (num_windows, window_size, 3)

                    # スケーリング適用（加速度のみ）
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        windowed_data = windowed_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # 形状を変換: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
                    windowed_data = np.transpose(windowed_data, (0, 2, 1))

                    # float16に変換
                    windowed_data = windowed_data.astype(np.float16)

                    # センサー/モダリティの階層構造
                    sensor_modality_key = f"{sensor_name}/{modality_name}"

                    processed[person_id][sensor_modality_key] = {
                        'X': windowed_data,
                        'Y': windowed_labels
                    }

                    logger.info(
                        f"  {sensor_modality_key}: X.shape={windowed_data.shape}, "
                        f"Y.shape={windowed_labels.shape}"
                    )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        処理済みデータを保存

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        保存形式:
            data/processed/opportunity/USER00001/BACK/ACC/X.npy, Y.npy
            data/processed/opportunity/USER00001/BACK/GYRO/X.npy, Y.npy
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
