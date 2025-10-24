"""
DSADS (Daily and Sports Activities Dataset) 前処理

DSADS データセット:
- 19種類の日常動作とスポーツ動作
- 8人の被験者
- 5つのIMUセンサー（45チャンネル）
- サンプリングレート: 25Hz
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import shutil

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    split_train_val_test,
    save_npy_dataset,
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


# DSADS データセットのURL
DSADS_URL = "https://archive.ics.uci.edu/static/public/256/daily+and+sports+activities.zip"


@register_preprocessor('dsads')
class DSADSPreprocessor(BasePreprocessor):
    """
    DSADSデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # DSADS固有の設定
        self.num_activities = 19
        self.num_subjects = 8
        self.num_sensors = 5
        self.channels_per_sensor = 9  # 3-axis acc, gyro, mag
        self.num_channels = 45  # 5 sensors × 9 channels

        # サンプリングレート
        self.original_sampling_rate = 25  # Hz (DSADSのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名とチャンネルマッピング
        self.sensor_names = ['Torso', 'RightArm', 'LeftArm', 'RightLeg', 'LeftLeg']
        self.sensor_channel_ranges = {
            'Torso': (0, 9),      # channels 0-8
            'RightArm': (9, 18),  # channels 9-17
            'LeftArm': (18, 27),  # channels 18-26
            'RightLeg': (27, 36), # channels 27-35
            'LeftLeg': (36, 45)   # channels 36-44
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.modalities = ['ACC', 'GYRO', 'MAG']
        self.modality_channel_ranges = {
            'ACC': (0, 3),   # 3軸加速度
            'GYRO': (3, 6),  # 3軸ジャイロ
            'MAG': (6, 9)    # 3軸地磁気
        }
        self.channels_per_modality = 3

        # 前処理パラメータ
        self.window_size = config.get('window_size', 125)  # 5秒 @ 25Hz
        self.stride = config.get('stride', 25)  # 1秒 @ 25Hz

        # スケーリング係数（m/s^2 -> G に変換）
        self.scale_factor = DATASETS.get('DSADS', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'dsads'

    def download_dataset(self) -> None:
        """
        DSADSデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading DSADS dataset")
        logger.info("=" * 80)

        dsads_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(dsads_raw_path, required_files=['a*/p*/s*.txt']):
            logger.warning(f"DSADS data already exists at {dsads_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive")
            dsads_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = dsads_raw_path.parent / 'dsads.zip'
            download_file(DSADS_URL, zip_path, desc='Downloading DSADS')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = dsads_raw_path.parent / 'dsads_temp'
            extract_archive(zip_path, extract_to, desc='Extracting DSADS')
            self._organize_dsads_data(extract_to, dsads_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: DSADS dataset downloaded to {dsads_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download DSADS dataset: {e}", exc_info=True)
            raise

    def _organize_dsads_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        DSADSデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/dsads）
        """
        logger.info(f"Organizing DSADS data from {extracted_path} to {target_path}")

        # 解凍されたデータのルートを見つける
        data_root = extracted_path

        # "data" フォルダを探す
        if (extracted_path / "data").exists():
            data_root = extracted_path / "data"
        elif (extracted_path / "Daily and Sports Activities").exists():
            data_root = extracted_path / "Daily and Sports Activities"

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find data directory in {extracted_path}")

        # a01, a02, ... などのディレクトリを探してコピー
        activity_dirs = sorted(data_root.glob("a*"))

        if not activity_dirs:
            # 別の構造を試す - 全内容をコピー
            logger.warning("Standard structure not found, copying all contents...")
            for item in data_root.iterdir():
                target_item = target_path / item.name
                if target_item.exists():
                    if target_item.is_dir():
                        shutil.rmtree(target_item)
                    else:
                        target_item.unlink()
                if item.is_dir():
                    shutil.copytree(item, target_item)
                else:
                    shutil.copy2(item, target_item)
        else:
            # 標準構造
            from tqdm import tqdm
            for activity_dir in tqdm(activity_dirs, desc='Organizing activities'):
                activity_name = activity_dir.name
                target_activity_dir = target_path / activity_name

                # アクティビティディレクトリをコピー
                if target_activity_dir.exists():
                    shutil.rmtree(target_activity_dir)
                shutil.copytree(activity_dir, target_activity_dir)

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        DSADSの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/dsads/a01/p1/s01.txt (activity 1, person 1, segment 1)
        - 各ファイル: (samples, 45) のテキストファイル

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 45) の配列
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"DSADS raw data not found at {raw_path}\n"
                "Expected structure: data/raw/dsads/a01/p1/s01.txt"
            )

        # 被験者ごとにデータを格納
        person_data = {person_id: {'data': [], 'labels': []}
                       for person_id in range(1, self.num_subjects + 1)}

        # 各活動について
        for activity_id in range(1, self.num_activities + 1):
            activity_dir = raw_path / f"a{activity_id:02d}"

            if not activity_dir.exists():
                logger.warning(f"Activity directory not found: {activity_dir}")
                continue

            # 各被験者について
            for person_id in range(1, self.num_subjects + 1):
                person_dir = activity_dir / f"p{person_id}"

                if not person_dir.exists():
                    logger.warning(f"Person directory not found: {person_dir}")
                    continue

                # 各セグメントについて
                segment_files = sorted(person_dir.glob("s*.txt"))

                for segment_file in segment_files:
                    try:
                        # データ読み込み
                        segment_data = np.loadtxt(segment_file, delimiter=',')

                        # データ形状チェック
                        if segment_data.shape[1] != self.num_channels:
                            logger.warning(
                                f"Unexpected number of channels in {segment_file}: "
                                f"{segment_data.shape[1]} (expected {self.num_channels})"
                            )
                            continue

                        # ラベル生成（0-indexed）
                        segment_labels = np.full(len(segment_data), activity_id - 1)

                        person_data[person_id]['data'].append(segment_data)
                        person_data[person_id]['labels'].append(segment_labels)

                    except Exception as e:
                        logger.error(f"Error loading {segment_file}: {e}")
                        continue

        # 各被験者のデータを結合
        result = {}
        for person_id in range(1, self.num_subjects + 1):
            if person_data[person_id]['data']:
                data = np.vstack(person_data[person_id]['data'])
                labels = np.hstack(person_data[person_id]['labels'])
                result[person_id] = (data, labels)
                logger.info(f"USER{person_id:05d}: {data.shape}, Labels: {labels.shape}")
            else:
                logger.warning(f"No data loaded for USER{person_id:05d}")

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

            # リサンプリング (25Hz -> 30Hz)
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
        特徴抽出（センサー×モダリティごとのウィンドウ化と正規化）

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            例: {'Torso_ACC': {'X': (N, 3, 125), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # センサーのチャンネルを抽出
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]  # (samples, 9)

                # スライディングウィンドウ適用（最後のウィンドウはパディング）
                # 正規化は行わず、生のセンサーデータを保持
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # DSADS特別処理: 150に満たない場合はパディング
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
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

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
            data/processed/dsads/USER00001/Torso/ACC/X.npy, Y.npy
            data/processed/dsads/USER00001/Torso/GYRO/X.npy, Y.npy
            data/processed/dsads/USER00001/LeftArm/ACC/X.npy, Y.npy
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
