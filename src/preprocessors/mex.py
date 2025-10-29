"""
MEx (Multi-modal Exercise) データセット前処理

MEx データセット:
- 7種類の理学療法エクササイズ
- 30人の被験者
- 2つのAccelerometer（手首、太もも）
- サンプリングレート: 100Hz
- 加速度センサー（3軸、±8g、G単位）
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

# MEXデータセットのダウンロードURL
MEX_URL = "https://archive.ics.uci.edu/static/public/500/mex.zip"


@register_preprocessor('mex')
class MexPreprocessor(BasePreprocessor):
    """
    MEXデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # MEx固有の設定
        self.num_activities = 7
        self.num_subjects = 30
        self.num_sensors = 2
        self.num_channels = 6  # 2センサー × 3軸

        # センサー名とチャンネルマッピング
        # チャンネル構成:
        # Wrist: ACC(3) = 3
        # Thigh: ACC(3) = 3
        self.sensor_names = ['Wrist', 'Thigh']
        self.sensor_channel_ranges = {
            'Wrist': (0, 3),   # channels 0-2
            'Thigh': (3, 6)    # channels 3-5
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3)   # 3軸加速度
            },
            'Thigh': {
                'ACC': (0, 3)   # 3軸加速度
            }
        }

        # サンプリングレート
        self.original_sampling_rate = 100  # Hz (MEXのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（既にG単位なので不要）
        self.scale_factor = DATASETS.get('MEX', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'mex'

    def download_dataset(self) -> None:
        """
        MEXデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading MEX dataset")
        logger.info("=" * 80)

        dataset_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(dataset_path, required_files=['act/*/01_act_1.csv']):
            logger.warning(f"MEX data already exists at {dataset_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/3: Downloading archive")
            zip_path = self.raw_data_path / "mex.zip"
            download_file(MEX_URL, zip_path, desc='Downloading MEX')

            # 2. 外側のZIPファイルの解凍（一時ディレクトリに）
            logger.info("Step 2/3: Extracting outer archive")
            temp_dir = self.raw_data_path / "mex_temp"
            extract_archive(zip_path, temp_dir, desc='Extracting MEX outer archive')

            # 3. ネストされたdata.zipを確認して解凍
            logger.info("Step 3/3: Extracting nested data.zip")
            nested_zip = temp_dir / "data.zip"
            if nested_zip.exists():
                dataset_path.mkdir(parents=True, exist_ok=True)
                extract_archive(nested_zip, dataset_path, desc='Extracting MEX data')
                nested_zip.unlink()
            else:
                logger.error(f"data.zip not found in {temp_dir}")
                raise FileNotFoundError(f"Expected data.zip in {temp_dir}")

            # クリーンアップ
            if zip_path.exists():
                zip_path.unlink()
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)

            logger.info(f"MEX dataset downloaded and extracted to {dataset_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download MEX dataset: {e}")
            raise

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        MEXの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/mex/MEx/acw/{subject_id}/{exercise_id}.txt (wrist accelerometer)
        - data/raw/mex/MEx/act/{subject_id}/{exercise_id}.txt (thigh accelerometer)
        - 各ファイル: timestamp x y z (スペース区切り)
        - エクササイズID: 1-7 (エクササイズ4は左右2回: 4L.txt, 4R.txt)
        - ラベル: 0-6 (0-indexed)

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 6) の配列 [wrist_xyz, thigh_xyz]
                labels: (num_samples,) の配列（0-indexed）
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"MEx raw data not found at {raw_path}\n"
                "Expected structure: data/raw/mex/acw/, data/raw/mex/act/"
            )

        acw_path = raw_path / "acw"  # wrist accelerometer
        act_path = raw_path / "act"  # thigh accelerometer

        if not acw_path.exists() or not act_path.exists():
            raise FileNotFoundError(
                f"MEx sensor folders not found\n"
                f"Expected: {acw_path} and {act_path}"
            )

        # 被験者ごとにデータを格納
        result = {}

        # 被験者IDは1-30（フォルダ名は01, 02, ... 30で0パディング）
        for subject_id in range(1, 31):
            subject_str = f"{subject_id:02d}"  # 01, 02, ... 30
            subject_acw_dir = acw_path / subject_str
            subject_act_dir = act_path / subject_str

            if not subject_acw_dir.exists() or not subject_act_dir.exists():
                logger.warning(f"Subject {subject_id} folders not found, skipping")
                continue

            # 各被験者の全エクササイズデータを統合
            all_wrist_data = []
            all_thigh_data = []
            all_labels = []

            # エクササイズファイルをロード (1-7)
            # ファイル名形式: {subject_id}_act_{ex_id}.csv と {subject_id}_acw_{ex_id}.csv
            # エクササイズ4は2ファイル（04_act_1.csv, 04_act_2.csv）
            exercise_files = []
            for ex_id in range(1, 8):
                if ex_id == 4:
                    # エクササイズ4は2回実施（_1と_2）
                    exercise_files.append((f"{ex_id:02d}_act_1.csv", f"{ex_id:02d}_acw_1.csv", ex_id - 1))
                    exercise_files.append((f"{ex_id:02d}_act_2.csv", f"{ex_id:02d}_acw_2.csv", ex_id - 1))
                else:
                    exercise_files.append((f"{ex_id:02d}_act_1.csv", f"{ex_id:02d}_acw_1.csv", ex_id - 1))

            for thigh_filename, wrist_filename, label in exercise_files:
                wrist_file = subject_acw_dir / wrist_filename
                thigh_file = subject_act_dir / thigh_filename

                if not wrist_file.exists() or not thigh_file.exists():
                    logger.debug(f"Subject {subject_id}, files {wrist_filename}/{thigh_filename} not found, skipping")
                    continue

                try:
                    # 手首加速度データ読み込み（カンマ区切り、タイムスタンプ,x,y,z）
                    wrist_df = pd.read_csv(wrist_file, header=None, names=['timestamp', 'x', 'y', 'z'])
                    wrist_data = wrist_df[['x', 'y', 'z']].values.astype(np.float32)

                    # 太もも加速度データ読み込み（カンマ区切り、タイムスタンプ,x,y,z）
                    thigh_df = pd.read_csv(thigh_file, header=None, names=['timestamp', 'x', 'y', 'z'])
                    thigh_data = thigh_df[['x', 'y', 'z']].values.astype(np.float32)

                    # サンプル数を揃える（短い方に合わせる）
                    min_samples = min(len(wrist_data), len(thigh_data))
                    wrist_data = wrist_data[:min_samples]
                    thigh_data = thigh_data[:min_samples]

                    # ラベル生成
                    exercise_labels = np.full(min_samples, label, dtype=int)

                    all_wrist_data.append(wrist_data)
                    all_thigh_data.append(thigh_data)
                    all_labels.append(exercise_labels)

                except Exception as e:
                    logger.error(f"Error loading subject {subject_id}, file {filename}: {e}")
                    continue

            if not all_wrist_data:
                logger.warning(f"No data loaded for subject {subject_id}")
                continue

            # 全エクササイズを結合
            wrist_data = np.vstack(all_wrist_data)  # (total_samples, 3)
            thigh_data = np.vstack(all_thigh_data)  # (total_samples, 3)
            labels = np.concatenate(all_labels)      # (total_samples,)

            # 手首と太ももを結合: (total_samples, 6)
            sensor_data = np.hstack([wrist_data, thigh_data])

            # person_idは1-indexed（USER00001から開始）
            person_id = subject_id
            result[person_id] = (sensor_data, labels)
            logger.info(
                f"USER{person_id:05d}: {sensor_data.shape}, Labels: {labels.shape}"
            )

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
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
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
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]

                # スライディングウィンドウ適用（最後のウィンドウはパディング）
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # MEx: 150に満たない場合はパディング
                )
                # windowed_data: (num_windows, window_size, sensor_channels)

                # 各モダリティに分割（MExはACCのみ）
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, channels)

                    # スケーリングは不要（既にG単位）
                    if self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # 形状を変換: (num_windows, window_size, C) -> (num_windows, C, window_size)
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
            data/processed/mex/USER00001/Wrist/ACC/X.npy, Y.npy
            data/processed/mex/USER00001/Thigh/ACC/X.npy, Y.npy
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
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',  # 正規化なし（生データ保持）
            'scale_factor': self.scale_factor,  # スケーリング係数（既にG単位なのでNone）
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
