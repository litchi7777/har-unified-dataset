"""
OPPORTUNITY (OPPORTUNITY Activity Recognition Dataset) 前処理

OPPORTUNITY データセット:
- 17種類のmid-level gestures (+ Null class)
- 4人の被験者
- 113チャンネルのBody-wornセンサー（7つのIMU + 12個の加速度センサー）
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


# 113センサーチャンネルを選択（DeepConvLSTMの実装に基づく）
# 元データの列0はタイムスタンプ、列1-133がbody-wornセンサー、列134-242がオブジェクト/アンビエントセンサー
# 列243がmid-level gestureラベル、列244がlocomotionラベル
def select_columns_opp(data):
    """
    OPPORTUNITYチャレンジで使用される113列を選択

    - 列46-49, 59-62, 72-75, 85-88, 98-101: IMUのQuaternion（削除）
    - 列134-242: オブジェクト/アンビエントセンサー（削除）
    - 列244-248: その他（削除）

    Args:
        data: 元のデータ行列 (samples, 249)

    Returns:
        選択された113列のセンサーデータ (samples, 113)
    """
    features_delete = np.arange(46, 50)
    features_delete = np.concatenate([features_delete, np.arange(59, 63)])
    features_delete = np.concatenate([features_delete, np.arange(72, 76)])
    features_delete = np.concatenate([features_delete, np.arange(85, 89)])
    features_delete = np.concatenate([features_delete, np.arange(98, 102)])
    features_delete = np.concatenate([features_delete, np.arange(134, 243)])
    features_delete = np.concatenate([features_delete, np.arange(244, 249)])
    return np.delete(data, features_delete, 1)


# センサーチャンネルのグループ定義（113チャンネル）
# 各IMUは17チャンネル - 4チャンネル(QUAT) = 13チャンネル
# 列番号は選択後の0-indexedインデックス
SENSOR_GROUPS = {
    'BACK': {
        'channels': list(range(0, 13)),  # 13チャンネル
        'modalities': {
            'ACC': list(range(0, 3)),    # 加速度 X, Y, Z
            'GYRO': list(range(3, 6)),   # ジャイロ X, Y, Z
            'MAG': list(range(6, 9)),    # 地磁気 X, Y, Z
        }
    },
    'RUA': {  # Right Upper Arm
        'channels': list(range(13, 26)),
        'modalities': {
            'ACC': list(range(13, 16)),
            'GYRO': list(range(16, 19)),
            'MAG': list(range(19, 22)),
        }
    },
    'RLA': {  # Right Lower Arm
        'channels': list(range(26, 39)),
        'modalities': {
            'ACC': list(range(26, 29)),
            'GYRO': list(range(29, 32)),
            'MAG': list(range(32, 35)),
        }
    },
    'LUA': {  # Left Upper Arm
        'channels': list(range(39, 52)),
        'modalities': {
            'ACC': list(range(39, 42)),
            'GYRO': list(range(42, 45)),
            'MAG': list(range(45, 48)),
        }
    },
    'LLA': {  # Left Lower Arm
        'channels': list(range(52, 65)),
        'modalities': {
            'ACC': list(range(52, 55)),
            'GYRO': list(range(55, 58)),
            'MAG': list(range(58, 61)),
        }
    },
    'L_SHOE': {  # Left Shoe
        'channels': list(range(65, 78)),
        'modalities': {
            'ACC': list(range(65, 68)),
            'GYRO': list(range(68, 71)),
            'MAG': list(range(71, 74)),
        }
    },
    'R_SHOE': {  # Right Shoe
        'channels': list(range(78, 91)),
        'modalities': {
            'ACC': list(range(78, 81)),
            'GYRO': list(range(81, 84)),
            'MAG': list(range(84, 87)),
        }
    },
    # 残り22チャンネル（91-112）は12個の3D加速度センサー（各3チャンネル）
    # 位置: HIP, RKN (Right Knee), LKNなど
    'ACC_SENSORS': {
        'channels': list(range(91, 113)),  # 22チャンネル（残りの加速度センサー）
        'modalities': {
            'ACC': list(range(91, 113)),  # 全て加速度センサー
        }
    }
}

# ラベル列（mid-level gestures）
LABEL_COLUMN = 243  # 元データでの列番号


@register_preprocessor('opportunity')
class OpportunityPreprocessor(BasePreprocessor):
    """
    OPPORTUNITYデータセット用の前処理クラス（113チャンネル全センサー使用）
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # OPPORTUNITY固有の設定
        self.num_activities = 17  # mid-level gestures
        self.num_subjects = 4
        self.num_channels = 113  # 全body-wornセンサー

        # サンプリングレート
        self.original_sampling_rate = 30  # Hz (OPPORTUNITYのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサーグループ
        self.sensor_groups = SENSOR_GROUPS
        self.sensor_names = list(SENSOR_GROUPS.keys())

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
        - 各ファイル: (samples, 249) のテキストファイル（スペース区切り）

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 113) の配列（選択されたセンサー列）
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
                    # データ読み込み（スペース区切り）
                    data = np.loadtxt(data_file, dtype=np.float32)

                    if data.ndim == 1:
                        data = data.reshape(1, -1)

                    logger.info(f"  Loaded {data_file.name}: {data.shape}")

                    # 列選択（113チャンネルを抽出）
                    selected_data = select_columns_opp(data)

                    # ラベル抽出（mid-level gestures: 選択後の列113、元は列243）
                    # select_columns_oppは列0を保持し、列1-113がセンサー、列114がラベル
                    labels = selected_data[:, 113].astype(np.int32)

                    # センサーデータのみ抽出（列1-113）
                    sensor_data = selected_data[:, 1:114]

                    # ラベル変換: 0 -> -1 (Null class), その他は調整
                    # gestures label adjustment (DeepConvLSTMの実装に基づく)
                    label_map = {
                        0: -1,       # Null -> -1
                        406516: 0,   # Open Door 1
                        406517: 1,   # Open Door 2
                        404516: 2,   # Close Door 1
                        404517: 3,   # Close Door 2
                        406520: 4,   # Open Fridge
                        404520: 5,   # Close Fridge
                        406505: 6,   # Open Dishwasher
                        404505: 7,   # Close Dishwasher
                        406519: 8,   # Open Drawer 1
                        404519: 9,   # Close Drawer 1
                        406511: 10,  # Open Drawer 2
                        404511: 11,  # Close Drawer 2
                        406508: 12,  # Open Drawer 3
                        404508: 13,  # Close Drawer 3
                        408512: 14,  # Clean Table
                        407521: 15,  # Drink from Cup
                        405506: 16,  # Toggle Switch
                    }

                    for old_label, new_label in label_map.items():
                        labels[labels == old_label] = new_label

                    person_data[person_id]['data'].append(sensor_data)
                    person_data[person_id]['labels'].append(labels)

                except Exception as e:
                    logger.error(f"Error loading {data_file}: {e}")
                    import traceback
                    traceback.print_exc()
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
        特徴抽出（センサーグループ×モダリティごとのウィンドウ化とスケーリング）

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーグループについて処理
            for sensor_name, sensor_info in self.sensor_groups.items():
                for modality_name, modality_channels in sensor_info['modalities'].items():
                    # 該当モダリティのチャンネルを抽出
                    sensor_modality_data = person_data[:, modality_channels]

                    # スライディングウィンドウ適用
                    windowed_data, windowed_labels = create_sliding_windows(
                        sensor_modality_data,
                        labels,
                        window_size=self.window_size,
                        stride=self.stride,
                        drop_last=False,
                        pad_last=True
                    )
                    # windowed_data: (num_windows, window_size, num_channels)

                    # スケーリング適用（加速度のみ）
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        windowed_data = windowed_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # 形状を変換: (num_windows, window_size, channels) -> (num_windows, channels, window_size)
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
            'num_channels': self.num_channels,
            'sensor_groups': list(self.sensor_groups.keys()),
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
