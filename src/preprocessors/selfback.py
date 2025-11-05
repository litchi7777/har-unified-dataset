"""
SelfBack (Self-management support for low back pain) 前処理

SelfBack データセット:
- 9種類の日常動作（歩行3種、階段2種、ジョギング、座る、立つ、寝る）
- 33人の被験者
- 2つの加速度センサー（各3軸）: 手首(Wrist) + 太もも(Thigh)
- サンプリングレート: 100Hz
- 加速度単位: G（±8g範囲、すでにG単位）
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


# SelfBack データセットのURL
SELFBACK_URL = "https://archive.ics.uci.edu/static/public/521/selfback.zip"


@register_preprocessor('selfback')
class SelfBackPreprocessor(BasePreprocessor):
    """
    SelfBackデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # SelfBack固有の設定
        self.num_activities = 9
        self.num_subjects = 33
        self.num_sensors = 2  # Wrist + Thigh
        self.channels_per_sensor = 3  # 3-axis acc
        self.num_channels = 6  # 2 sensors × 3 channels

        # サンプリングレート
        self.original_sampling_rate = 100  # Hz (SelfBackのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # センサー名とチャンネルマッピング
        self.sensor_names = ['Wrist', 'Thigh']
        self.sensor_channel_ranges = {
            'Wrist': (0, 3),  # channels 0-2
            'Thigh': (3, 6),  # channels 3-5
        }

        # モダリティ（各センサーはACCのみ）
        self.modalities = ['ACC']
        self.modality_channel_ranges = {
            'ACC': (0, 3),  # 3軸加速度
        }
        self.channels_per_modality = 3

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（SelfBackはすでにG単位なのでNone）
        self.scale_factor = DATASETS.get('SELFBACK', {}).get('scale_factor', None)

        # アクティビティ名マッピング
        self.activity_names = {
            'downstairs': 0,
            'upstairs': 1,
            'walk_slow': 2,
            'walk_mod': 3,
            'walk_fast': 4,
            'jogging': 5,
            'sitting': 6,
            'standing': 7,
            'lying': 8,
        }

        # ユーザーID再マッピング用（load_raw_dataで初期化）
        self.user_id_mapping = {}

    def get_dataset_name(self) -> str:
        return 'selfback'

    def download_dataset(self) -> None:
        """
        SelfBackデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading SelfBack dataset")
        logger.info("=" * 80)

        selfback_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(selfback_raw_path, required_files=['w/*/*.csv', 't/*/*.csv']):
            logger.warning(f"SelfBack data already exists at {selfback_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive")
            selfback_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = selfback_raw_path.parent / 'selfback.zip'
            download_file(SELFBACK_URL, zip_path, desc='Downloading SelfBack')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = selfback_raw_path.parent / 'selfback_temp'
            extract_archive(zip_path, extract_to, desc='Extracting SelfBack')
            self._organize_selfback_data(extract_to, selfback_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: SelfBack dataset downloaded to {selfback_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download SelfBack dataset: {e}", exc_info=True)
            raise

    def _organize_selfback_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        SelfBackデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/selfback）
        """
        import shutil
        from tqdm import tqdm

        logger.info(f"Organizing SelfBack data from {extracted_path} to {target_path}")

        # selfBACKフォルダを探す
        selfback_root = extracted_path / "selfBACK"
        if not selfback_root.exists():
            # 直接抽出された場合
            selfback_root = extracted_path

        if not selfback_root.exists():
            raise FileNotFoundError(f"Could not find selfBACK directory in {extracted_path}")

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        # w, t ディレクトリをコピー
        for sensor_dir in ['w', 't']:
            source_dir = selfback_root / sensor_dir
            if source_dir.exists():
                target_sensor_dir = target_path / sensor_dir
                if target_sensor_dir.exists():
                    shutil.rmtree(target_sensor_dir)
                shutil.copytree(source_dir, target_sensor_dir)
                logger.info(f"Copied {sensor_dir}/ directory")

        logger.info(f"Data organized at: {target_path}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        SelfBackの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/selfback/w/{activity}/{subject_id}.csv (wrist sensor)
        - data/raw/selfback/t/{activity}/{subject_id}.csv (thigh sensor)
        - 各ファイル: time,x,y,z (CSVヘッダー付き)

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 6) の配列 [wrist_x, wrist_y, wrist_z, thigh_x, thigh_y, thigh_z]
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"SelfBack raw data not found at {raw_path}\n"
                "Expected structure: data/raw/selfback/w/{activity}/{subject_id}.csv"
            )

        # 被験者IDを抽出（wディレクトリから）
        w_dir = raw_path / 'w'
        if not w_dir.exists():
            raise FileNotFoundError(f"Wrist sensor directory not found: {w_dir}")

        # 最初のアクティビティディレクトリから被験者IDリストを取得
        first_activity = list(w_dir.iterdir())[0]
        subject_files = [f for f in first_activity.glob("*.csv") if not f.name.startswith('._')]
        original_subject_ids = sorted([int(f.stem) for f in subject_files])

        logger.info(f"Found {len(original_subject_ids)} subjects: {min(original_subject_ids)}-{max(original_subject_ids)}")

        # ユーザーIDマッピングを作成（元のID → シーケンシャルID）
        # 例: 26 -> 1, 27 -> 2, ..., 63 -> 33
        self.user_id_mapping = {
            original_id: new_id
            for new_id, original_id in enumerate(original_subject_ids, start=1)
        }
        logger.info(f"User ID mapping created: {min(original_subject_ids)} -> 1, {max(original_subject_ids)} -> {len(original_subject_ids)}")

        # 被験者ごとにデータを格納（マップされたIDを使用）
        person_data = {}

        for original_id in original_subject_ids:
            mapped_id = self.user_id_mapping[original_id]
            person_data[mapped_id] = {'data': [], 'labels': []}

            # 各アクティビティについて
            for activity_name, activity_id in self.activity_names.items():
                w_file = w_dir / activity_name / f"{original_id:03d}.csv"
                t_file = (raw_path / 't') / activity_name / f"{original_id:03d}.csv"

                if not w_file.exists() or not t_file.exists():
                    logger.warning(f"Missing files for USER{mapped_id:05d} (original ID: {original_id}), activity {activity_name}")
                    continue

                try:
                    # 手首センサーデータ読み込み
                    w_data = pd.read_csv(w_file)
                    # 太ももセンサーデータ読み込み
                    t_data = pd.read_csv(t_file)

                    # タイムスタンプを除いてx,y,z列のみ抽出
                    w_values = w_data[['x', 'y', 'z']].values
                    t_values = t_data[['x', 'y', 'z']].values

                    # 長さを合わせる（短い方に合わせる）
                    min_len = min(len(w_values), len(t_values))
                    w_values = w_values[:min_len]
                    t_values = t_values[:min_len]

                    # マージ: [wrist_x, wrist_y, wrist_z, thigh_x, thigh_y, thigh_z]
                    merged_data = np.hstack([w_values, t_values])  # (N, 6)

                    # ラベル生成
                    labels = np.full(len(merged_data), activity_id)

                    person_data[mapped_id]['data'].append(merged_data)
                    person_data[mapped_id]['labels'].append(labels)

                except Exception as e:
                    logger.error(f"Error loading USER{mapped_id:05d} (original ID: {original_id}), activity {activity_name}: {e}")
                    continue

        # 各被験者のデータを結合
        result = {}
        for original_id in original_subject_ids:
            mapped_id = self.user_id_mapping[original_id]
            if person_data[mapped_id]['data']:
                data = np.vstack(person_data[mapped_id]['data'])
                labels = np.hstack(person_data[mapped_id]['labels'])
                result[mapped_id] = (data, labels)
                logger.info(f"USER{mapped_id:05d} (original ID: {original_id}): {data.shape}, Labels: {labels.shape}")
            else:
                logger.warning(f"No data loaded for USER{mapped_id:05d} (original ID: {original_id})")

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
            例: {'Wrist/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]

                # センサーのチャンネルを抽出
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]  # (samples, 3)

                # スライディングウィンドウ適用
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True
                )
                # windowed_data: (num_windows, window_size, 3)

                # 各モダリティに分割（SelfBackはACCのみ）
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, 3)

                    # スケーリング適用（SelfBackはすでにG単位なのでscale_factor=None）
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
            data/processed/selfback/USER00001/Wrist/ACC/X.npy, Y.npy
            data/processed/selfback/USER00001/Thigh/ACC/X.npy, Y.npy
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
            'scale_factor': self.scale_factor,  # SelfBackはすでにG単位なのでNone
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
