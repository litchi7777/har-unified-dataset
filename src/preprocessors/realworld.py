"""
RealWorld データセット前処理

RealWorld データセット (2016):
- 8種類の身体活動
- 15人の被験者
- 7つのセンサー位置（Chest, Forearm, Head, Shin, Thigh, UpperArm, Waist）
- サンプリングレート: 50Hz
- 3軸加速度、ジャイロ、磁力計センサー
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
import zipfile
import requests
from tqdm import tqdm

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


@register_preprocessor('realworld')
class RealWorldPreprocessor(BasePreprocessor):
    """
    RealWorldデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # RealWorld固有の設定
        self.num_activities = 8
        self.num_subjects = 15
        self.num_sensors = 7

        # センサー名（RealWorldの命名規則に合わせる）
        self.sensor_names = ['Chest', 'Forearm', 'Head', 'Shin', 'Thigh', 'UpperArm', 'Waist']

        # モダリティ
        self.modality_names = ['ACC', 'GYRO', 'MAG']

        # 各モダリティは3軸（x, y, z）
        self.channels_per_modality = 3

        # サンプリングレート
        self.original_sampling_rate = 50  # Hz (RealWorldのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（データを確認後に決定）
        self.scale_factor = DATASETS.get('REALWORLD', {}).get('scale_factor', None)

        # 活動名マッピング（ディレクトリ名 → ラベルID）
        # RealWorldデータセットのディレクトリ名は小文字
        self.activity_mapping = {
            'climbingdown': 0,
            'climbingup': 1,
            'jumping': 2,
            'lying': 3,
            'running': 4,
            'sitting': 5,
            'standing': 6,
            'walking': 7
        }

        # ダウンロードURL
        self.download_url = "http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip"

    def get_dataset_name(self) -> str:
        return 'realworld'

    def download_dataset(self) -> None:
        """
        RealWorldデータセットをダウンロード
        """
        download_path = self.raw_data_path / f"{self.dataset_name}.zip"
        extract_path = self.raw_data_path / self.dataset_name

        # 既にダウンロード済みかチェック
        if extract_path.exists() and any(extract_path.iterdir()):
            logger.info(f"Dataset already exists at {extract_path}")
            return

        # ダウンロード
        logger.info(f"Downloading RealWorld dataset from {self.download_url}")
        try:
            response = requests.get(self.download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(download_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading') as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"Downloaded to {download_path}")

            # 解凍
            logger.info(f"Extracting {download_path} to {extract_path}")
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_path)

            logger.info("Download and extraction completed")

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    def _load_sensor_data(self, proband_path: Path, sensor: str, modality: str) -> pd.DataFrame:
        """
        特定のセンサー×モダリティのCSVデータを読み込む

        Args:
            proband_path: 被験者のディレクトリパス
            sensor: センサー名（例: 'chest'）
            modality: モダリティ名（例: 'acc'）

        Returns:
            センサーデータのDataFrame（columns: timestamp, x, y, z）
        """
        # RealWorldのファイル名は小文字
        sensor_lower = sensor.lower()
        modality_lower = modality.lower()

        # ファイルパス例: proband1/data/acc_chest_csv/
        data_dir = proband_path / 'data' / f'{modality_lower}_{sensor_lower}_csv'

        # zipファイルの場合
        if not data_dir.exists():
            zip_path = proband_path / 'data' / f'{modality_lower}_{sensor_lower}_csv.zip'
            if zip_path.exists():
                # 一時的に解凍
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(proband_path / 'data')

        if not data_dir.exists():
            logger.warning(f"Sensor data not found: {data_dir}")
            return None

        # CSVファイルを読み込み（複数ファイルがある場合は結合）
        csv_files = sorted(data_dir.glob('*.csv'))

        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return None

        # 全CSVファイルを結合
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # 活動名を抽出（ファイル名に含まれている）
                # 例: acc_climbingdown_csv.csv -> climbingdown
                activity = csv_file.stem.split('_')[1] if '_' in csv_file.stem else None
                if activity:
                    df['activity'] = activity
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading {csv_file}: {e}")
                continue

        if not dfs:
            return None

        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df

    def load_raw_data(self) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        RealWorldの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/realworld/realworld2016_dataset/proband1/
        - data/raw/realworld/realworld2016_dataset/proband2/

        Returns:
            {person_id: {sensor: {modality: (data, labels)}}}
        """
        # データセットのルートパス
        dataset_root = self.raw_data_path / self.dataset_name / 'realworld2016_dataset'

        if not dataset_root.exists():
            # realworld2016フォルダ直下にprobandがある場合
            dataset_root = self.raw_data_path / self.dataset_name

        if not dataset_root.exists():
            raise FileNotFoundError(
                f"RealWorld raw data not found at {dataset_root}\n"
                "Expected structure: data/raw/realworld/realworld2016_dataset/proband1/"
            )

        result = {}

        # proband1～proband15を読み込み
        for person_id in range(1, self.num_subjects + 1):
            proband_path = dataset_root / f'proband{person_id}'

            if not proband_path.exists():
                logger.warning(f"Proband {person_id} not found at {proband_path}")
                continue

            logger.info(f"Loading USER{person_id:05d} from {proband_path.name}")

            result[person_id] = {}

            # 各センサーについて処理
            for sensor in self.sensor_names:
                result[person_id][sensor] = {}

                # 各モダリティについて処理
                for modality in self.modality_names:
                    df = self._load_sensor_data(proband_path, sensor, modality)

                    if df is None or len(df) == 0:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}/{modality}: No data loaded"
                        )
                        continue

                    # センサーデータ抽出（x, y, z列）
                    # RealWorldのCSV: attr_time, attr_x, attr_y, attr_z
                    # または: timestamp, x, y, z
                    xyz_columns = [col for col in df.columns if col in ['attr_x', 'attr_y', 'attr_z', 'x', 'y', 'z']]

                    if len(xyz_columns) < 3:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}/{modality}: "
                            f"Expected 3 xyz columns, found {len(xyz_columns)}"
                        )
                        continue

                    # x, y, z データを抽出
                    sensor_data = df[xyz_columns[:3]].values.astype(np.float32)

                    # ラベル抽出（activityカラムから）
                    if 'activity' in df.columns:
                        labels = df['activity'].map(self.activity_mapping).values
                        # NaNを-1に変換（未定義活動）
                        labels = np.where(np.isnan(labels), -1, labels).astype(int)
                    else:
                        # activityカラムがない場合は全て-1
                        labels = np.full(len(sensor_data), -1, dtype=int)

                    result[person_id][sensor][modality] = (sensor_data, labels)

                    logger.info(
                        f"  {sensor}/{modality}: {sensor_data.shape}, "
                        f"Labels: {np.unique(labels)}"
                    )

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        データのクリーニングとリサンプリング

        Args:
            data: {person_id: {sensor: {modality: (data, labels)}}}

        Returns:
            クリーニング・リサンプリング済みデータ
        """
        cleaned = {}

        for person_id, sensor_dict in data.items():
            cleaned[person_id] = {}

            for sensor, modality_dict in sensor_dict.items():
                cleaned[person_id][sensor] = {}

                for modality, (sensor_data, labels) in modality_dict.items():
                    # 無効なサンプルを除去
                    cleaned_data, cleaned_labels = filter_invalid_samples(sensor_data, labels)

                    # 未定義ラベル（-1）を除外
                    valid_mask = cleaned_labels >= 0
                    cleaned_data = cleaned_data[valid_mask]
                    cleaned_labels = cleaned_labels[valid_mask]

                    if len(cleaned_data) == 0:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}/{modality}: "
                            "No valid data after filtering"
                        )
                        continue

                    # リサンプリング (50Hz -> 30Hz)
                    if self.original_sampling_rate != self.target_sampling_rate:
                        resampled_data, resampled_labels = resample_timeseries(
                            cleaned_data,
                            cleaned_labels,
                            self.original_sampling_rate,
                            self.target_sampling_rate
                        )
                        cleaned[person_id][sensor][modality] = (resampled_data, resampled_labels)
                        logger.info(
                            f"USER{person_id:05d}/{sensor}/{modality} "
                            f"resampled: {resampled_data.shape}"
                        )
                    else:
                        cleaned[person_id][sensor][modality] = (cleaned_data, cleaned_labels)
                        logger.info(
                            f"USER{person_id:05d}/{sensor}/{modality} "
                            f"cleaned: {cleaned_data.shape}"
                        )

        return cleaned

    def extract_features(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化）

        Args:
            data: {person_id: {sensor: {modality: (data, labels)}}}

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for person_id, sensor_dict in data.items():
            logger.info(f"Processing USER{person_id:05d}")
            processed[person_id] = {}

            for sensor, modality_dict in sensor_dict.items():
                for modality, (sensor_data, labels) in modality_dict.items():

                    if len(sensor_data) == 0:
                        continue

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

                    # スケーリング（必要に応じて）
                    if self.scale_factor is not None:
                        windowed_data = windowed_data / self.scale_factor
                        logger.info(
                            f"  Applied scale_factor={self.scale_factor} to "
                            f"{sensor}/{modality}"
                        )

                    # 形状を変換: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    windowed_data = np.transpose(windowed_data, (0, 2, 1))

                    # float16に変換
                    windowed_data = windowed_data.astype(np.float16)

                    # センサー/モダリティの階層構造
                    sensor_modality_key = f"{sensor}/{modality}"

                    processed[person_id][sensor_modality_key] = {
                        'X': windowed_data,
                        'Y': windowed_labels
                    }

                    logger.info(
                        f"  {sensor_modality_key}: X.shape={windowed_data.shape}, "
                        f"Y.shape={windowed_labels.shape}"
                    )

        return processed

    def save_processed_data(
        self,
        data: Dict[int, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        処理済みデータを保存

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        保存形式:
            data/processed/realworld/USER00001/Chest/ACC/X.npy, Y.npy
            data/processed/realworld/USER00001/Chest/GYRO/X.npy, Y.npy
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
            'modality_names': self.modality_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
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
                X = arrays['X']  # (num_windows, C, window_size)
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
