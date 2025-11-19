"""
HMP (Dataset for ADL Recognition with Wrist-worn Accelerometer) 前処理

HMP データセット:
- 14種類のADL（日常生活動作）
- 16人の被験者
- 1つの3軸加速度センサー（右手首）
- サンプリングレート: 32Hz
- 測定範囲: ±1.5g（6ビット分解能: 0-63）
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


# HMP データセットのURL
HMP_URL = "https://archive.ics.uci.edu/static/public/283/dataset+for+adl+recognition+with+wrist+worn+accelerometer.zip"


@register_preprocessor('hmp')
class HMPPreprocessor(BasePreprocessor):
    """
    HMPデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # HMP固有の設定
        self.num_activities = 14
        self.num_channels = 3  # 3軸加速度センサー

        # アクティビティマッピング（ディレクトリ名 -> ラベルID）
        self.activity_map = {
            'Brush_teeth': 0,
            'Climb_stairs': 1,
            'Comb_hair': 2,
            'Descend_stairs': 3,
            'Drink_glass': 4,
            'Eat_meat': 5,
            'Eat_soup': 6,
            'Getup_bed': 7,
            'Liedown_bed': 8,
            'Pour_water': 9,
            'Sitdown_chair': 10,
            'Standup_chair': 11,
            'Use_telephone': 12,
            'Walk': 13
        }

        # サンプリングレート
        self.original_sampling_rate = 32  # Hz (HMPのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名
        self.sensor_name = 'RightWrist'

        # モダリティ
        self.modality = 'ACC'

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（6ビット coded values: 0-63 = -1.5g to +1.5g -> G単位）
        # 変換式: real_val = -1.5g + (coded_val/63) * 3g
        # まず coded_val を -1.5 ~ +1.5 の範囲に変換してからG単位に
        self.scale_factor = DATASETS.get('HMP', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'hmp'

    def download_dataset(self) -> None:
        """
        HMPデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading HMP dataset")
        logger.info("=" * 80)

        hmp_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(hmp_raw_path, required_files=['*/Accelerometer-*.txt']):
            logger.warning(f"HMP data already exists at {hmp_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive")
            hmp_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = hmp_raw_path.parent / 'hmp.zip'
            download_file(HMP_URL, zip_path, desc='Downloading HMP')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = hmp_raw_path.parent / 'hmp_temp'
            extract_archive(zip_path, extract_to, desc='Extracting HMP')
            self._organize_hmp_data(extract_to, hmp_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: HMP dataset downloaded to {hmp_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download HMP dataset: {e}", exc_info=True)
            raise

    def _organize_hmp_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        HMPデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/hmp）
        """
        logger.info(f"Organizing HMP data from {extracted_path} to {target_path}")

        # 解凍されたデータのルートを見つける
        data_root = extracted_path

        # "HMP_Dataset" フォルダを探す
        if (extracted_path / "HMP_Dataset").exists():
            data_root = extracted_path / "HMP_Dataset"

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        if not data_root.exists():
            raise FileNotFoundError(f"Could not find HMP_Dataset directory in {extracted_path}")

        # アクティビティディレクトリを探してコピー（_MODEL以外）
        from tqdm import tqdm
        activity_dirs = [d for d in data_root.iterdir()
                        if d.is_dir() and not d.name.endswith('_MODEL')]

        if not activity_dirs:
            raise FileNotFoundError(f"No activity directories found in {data_root}")

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
        HMPの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/hmp/Brush_teeth/Accelerometer-2011-03-24-10-24-39-brush_teeth-f1.txt
        - 各ファイル: (samples, 3) のスペース区切りテキストファイル
        - 値は 0-63 の整数値（6ビット）

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 3) の配列
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"HMP raw data not found at {raw_path}\n"
                "Expected structure: data/raw/hmp/Brush_teeth/Accelerometer-*.txt"
            )

        # 被験者IDを収集（ファイル名から抽出）
        volunteer_ids = set()
        for activity_dir in raw_path.iterdir():
            if activity_dir.is_dir():
                for file_path in activity_dir.glob('Accelerometer-*.txt'):
                    # ファイル名から被験者IDを抽出
                    # 例: Accelerometer-2011-03-24-10-24-39-brush_teeth-f1.txt -> f1
                    filename_parts = file_path.stem.split('-')
                    if len(filename_parts) >= 9:
                        volunteer_id = filename_parts[-1]
                        volunteer_ids.add(volunteer_id)

        volunteer_ids = sorted(volunteer_ids)
        logger.info(f"Found {len(volunteer_ids)} volunteers: {volunteer_ids}")

        # 被験者IDを数値にマッピング（辞書順）
        volunteer_to_person_id = {vid: i+1 for i, vid in enumerate(volunteer_ids)}

        # 被験者ごとにデータを格納
        person_data = {person_id: {'data': [], 'labels': []}
                       for person_id in volunteer_to_person_id.values()}

        # 各活動について
        for activity_name, activity_id in self.activity_map.items():
            activity_dir = raw_path / activity_name

            if not activity_dir.exists():
                logger.warning(f"Activity directory not found: {activity_dir}")
                continue

            # 各ファイルについて
            txt_files = list(activity_dir.glob('Accelerometer-*.txt'))

            for file_path in txt_files:
                try:
                    # ファイル名から被験者IDを抽出
                    filename_parts = file_path.stem.split('-')
                    if len(filename_parts) < 9:
                        logger.warning(f"Unexpected filename format: {file_path.name}")
                        continue

                    volunteer_id = filename_parts[-1]
                    if volunteer_id not in volunteer_to_person_id:
                        logger.warning(f"Unknown volunteer ID: {volunteer_id}")
                        continue

                    person_id = volunteer_to_person_id[volunteer_id]

                    # データ読み込み（スペース区切り）
                    data = np.loadtxt(file_path)

                    # データ形状チェック
                    if data.ndim == 1:
                        # 1Dの場合は (samples, 3) に reshape
                        if len(data) % 3 == 0:
                            data = data.reshape(-1, 3)
                        else:
                            logger.warning(f"Cannot reshape {file_path.name}: {data.shape}")
                            continue

                    if data.shape[1] != self.num_channels:
                        logger.warning(
                            f"Unexpected number of channels in {file_path.name}: "
                            f"{data.shape[1]} (expected {self.num_channels})"
                        )
                        continue

                    # ラベル生成（全サンプルに同じラベル）
                    labels = np.full(len(data), activity_id, dtype=np.int32)

                    person_data[person_id]['data'].append(data)
                    person_data[person_id]['labels'].append(labels)

                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue

        # 各被験者のデータを結合
        result = {}
        for person_id in sorted(person_data.keys()):
            if person_data[person_id]['data']:
                data = np.vstack(person_data[person_id]['data'])
                labels = np.hstack(person_data[person_id]['labels'])
                result[person_id] = (data, labels)
                logger.info(f"USER{person_id:05d}: {data.shape}, Labels: {labels.shape}, "
                           f"Unique labels: {np.unique(labels)}")
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

            # リサンプリング (32Hz -> 30Hz)
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
        特徴抽出（ウィンドウ化とスケーリング）

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor_modality: {'X': data, 'Y': labels}}}
            例: {'RightWrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            # スライディングウィンドウ適用
            windowed_data, windowed_labels = create_sliding_windows(
                person_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )
            # windowed_data: (num_windows, window_size, 3)

            # スケーリング適用（6ビット coded values -> G単位）
            # 変換式: real_val = -1.5 + (coded_val/63) * 3.0
            # まず [0, 63] -> [-1.5, +1.5] に変換
            windowed_data = -1.5 + (windowed_data / 63.0) * 3.0
            logger.info(f"  Applied HMP-specific scaling: coded_val -> G")

            # 形状を変換: (num_windows, window_size, 3) -> (num_windows, 3, window_size)
            windowed_data = np.transpose(windowed_data, (0, 2, 1))

            # float16に変換
            windowed_data = windowed_data.astype(np.float16)

            # センサー/モダリティの階層構造
            sensor_modality_key = f"{self.sensor_name}/{self.modality}"

            processed[person_id] = {
                sensor_modality_key: {
                    'X': windowed_data,
                    'Y': windowed_labels
                }
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
            data/processed/hmp/USER00001/RightWrist/ACC/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_activities': self.num_activities,
            'sensor_names': [self.sensor_name],
            'modalities': [self.modality],
            'channels_per_modality': self.num_channels,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # 正規化なし（生データ保持）
            'scale_factor': 'HMP-specific: coded_val (0-63) -> G (-1.5 to +1.5)',
            'data_dtype': 'float16',
            'data_shape': f'(num_windows, {self.num_channels}, {self.window_size})',
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
