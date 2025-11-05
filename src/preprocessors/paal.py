"""
PAALデータセット用プリプロセッサ

データセット概要:
- 名称: PAAL ADL Accelerometry Dataset v2.0
- 出典: Zenodo (https://zenodo.org/records/5785955)
- 被験者: 52名（男性26名、女性26名、年齢18-77歳）
- 活動: 24種類のADL（日常生活動作）
- センサー: Empatica E4 加速度センサー（利き手に装着）
- サンプリングレート: 32Hz
- データ範囲: ±2g（8ビット分解能: 0.015g）
- ファイル形式: CSV（各ファイルは1つのアクティビティ1回の記録）
- ファイル命名: {activity}_{user_id}_{repetition}.csv
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .base import BasePreprocessor
from . import register_preprocessor
from .common import download_file, extract_archive, cleanup_temp_files, check_dataset_exists

logger = logging.getLogger(__name__)

# PAALデータセットのダウンロードURL
PAAL_URL = "https://zenodo.org/api/records/5785955/files-archive"


@register_preprocessor('paal')
class PAALPreprocessor(BasePreprocessor):
    """PAALデータセット用プリプロセッサ"""

    def __init__(self, config: Dict[str, Any]):
        """
        初期化

        Args:
            config: 設定辞書
        """
        super().__init__(config)

        # ウィンドウパラメータ（設定ファイルから取得）
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.overlap = config.get('overlap', 75)  # 50% オーバーラップ

        # データセット固有のパラメータ
        self.num_activities = 24
        self.num_subjects = 52
        self.num_sensors = 1  # Wrist (dominant hand)
        self.original_sampling_rate = 32  # Hz
        self.target_sampling_rate = 30  # Hz（統一サンプリングレート）
        self.scale_factor = 0.015  # 整数値からG単位への変換係数

        # 活動名のマッピング（ファイル名 → ラベルID）
        # アルファベット順にソート
        self.activity_names = {
            'blow_nose': 0,
            'brush_hair': 1,
            'brush_teeth': 2,
            'drink_water': 3,
            'dusting': 4,
            'eat_meal': 5,
            'ironing': 6,
            'open_a_bottle': 7,
            'open_a_box': 8,
            'phone_call': 9,
            'put_on_a_jacket': 10,
            'put_on_a_shoe': 11,
            'put_on_glasses': 12,
            'salute': 13,
            'sit_down': 14,
            'sneeze_cough': 15,
            'stand_up': 16,
            'take_off_a_jacket': 17,
            'take_off_a_shoe': 18,
            'take_off_glasses': 19,
            'type_on_a_keyboard': 20,
            'washing_dishes': 21,
            'washing_hands': 22,
            'writing': 23,
        }

    def get_dataset_name(self) -> str:
        """データセット名を返す"""
        return 'paal'

    def download_dataset(self) -> None:
        """
        PAALデータセットをダウンロードして解凍

        Zenodoから以下のファイルをダウンロード:
        - data.zip (加速度データ)
        - users.csv (ユーザー情報)
        - ADLs.csv (アクティビティ一覧)
        """
        logger.info("=" * 80)
        logger.info("Downloading PAAL dataset")
        logger.info("=" * 80)

        paal_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(paal_raw_path, required_files=['*.csv']):
            logger.warning(f"PAAL data already exists at {paal_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive from Zenodo")
            paal_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = paal_raw_path.parent / 'paal_data.zip'
            download_file(PAAL_URL, zip_path, desc='Downloading PAAL')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = paal_raw_path.parent / 'paal_temp'
            extract_archive(zip_path, extract_to, desc='Extracting PAAL')
            self._organize_paal_data(extract_to, paal_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: PAAL dataset downloaded to {paal_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download PAAL dataset: {e}", exc_info=True)
            raise

    def _organize_paal_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        PAALデータを適切なディレクトリ構造に整理

        解凍されたファイル:
        - data.zip (内部にdataset/*.csvが含まれる)
        - users.csv
        - ADLs.csv

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/paal）
        """
        import shutil
        import zipfile

        logger.info(f"Organizing PAAL data from {extracted_path} to {target_path}")

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        # data.zipを解凍
        data_zip = extracted_path / "data.zip"
        if data_zip.exists():
            logger.info("Extracting data.zip (accelerometer data)")
            with zipfile.ZipFile(data_zip, 'r') as zip_ref:
                zip_ref.extractall(extracted_path / "data_extracted")

            # dataset/*.csvをターゲットに移動
            dataset_dir = extracted_path / "data_extracted" / "dataset"
            if dataset_dir.exists():
                for csv_file in dataset_dir.glob("*.csv"):
                    # ._で始まるMacOSの隠しファイルをスキップ
                    if not csv_file.name.startswith('._'):
                        shutil.copy(csv_file, target_path / csv_file.name)
                logger.info(f"Copied {len(list(target_path.glob('*.csv')))} CSV files")
            else:
                raise FileNotFoundError(f"dataset directory not found in data.zip")

        # users.csvとADLs.csvをコピー（オプション：参照用）
        for meta_file in ['users.csv', 'ADLs.csv']:
            src = extracted_path / meta_file
            if src.exists():
                shutil.copy(src, target_path / meta_file)
                logger.debug(f"Copied {meta_file}")

        logger.info(f"Data organized at: {target_path}")

    def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        データのクリーニング

        PAALのケースでは、load_raw_data()で既に適切な形式に変換されているため、
        追加のクリーニングは不要。そのまま返す。

        Args:
            data: load_raw_data()の出力

        Returns:
            クリーニング済みデータ
        """
        # 無効なサンプル（データが空、または極端に短い）を除外
        valid_data = []
        for sample in data:
            if len(sample['data']) >= self.window_size:
                valid_data.append(sample)
            else:
                logger.debug(
                    f"Skipping short sample: subject={sample['subject_id']}, "
                    f"activity={sample['activity']}, length={len(sample['data'])}"
                )

        logger.info(f"Cleaned data: {len(valid_data)}/{len(data)} samples valid")
        return valid_data

    def load_raw_data(self) -> List[Dict[str, Any]]:
        """
        生データをロード

        Returns:
            データリスト。各要素は以下のキーを持つ辞書:
                - subject_id: 被験者ID (1-52)
                - activity: アクティビティID (0-23)
                - data: センサーデータ (N, 3) - X, Y, Z軸の加速度
        """
        raw_data_dir = Path(self.raw_data_path) / self.get_dataset_name()

        if not raw_data_dir.exists():
            raise FileNotFoundError(
                f"PAAL raw data not found at {raw_data_dir}. "
                f"Please download from: https://zenodo.org/records/5785955"
            )

        logger.info(f"Loading PAAL data from {raw_data_dir}")

        # データファイルをすべて読み込む
        data_files = sorted(raw_data_dir.glob("*.csv"))

        if len(data_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {raw_data_dir}. "
                f"Expected format: {{activity}}_{{user_id}}_{{repetition}}.csv"
            )

        logger.info(f"Found {len(data_files)} data files")

        all_data = []

        for file_path in data_files:
            # ファイル名から情報を抽出: {activity}_{user_id}_{repetition}.csv
            parts = file_path.stem.rsplit('_', 2)

            if len(parts) != 3:
                logger.warning(f"Skipping file with unexpected name format: {file_path.name}")
                continue

            activity_name = parts[0]
            user_id = int(parts[1])
            repetition = int(parts[2])

            # アクティビティ名をラベルIDに変換
            if activity_name not in self.activity_names:
                logger.warning(f"Unknown activity '{activity_name}' in file: {file_path.name}")
                continue

            activity_id = self.activity_names[activity_name]

            # CSVファイルを読み込む（ヘッダーなし、3列: x, y, z）
            try:
                sensor_data = pd.read_csv(
                    file_path,
                    header=None,
                    names=['x', 'y', 'z']
                ).values  # (N, 3)

                # 整数値からG単位に変換
                sensor_data = sensor_data * self.scale_factor

            except Exception as e:
                logger.error(f"Error reading file {file_path.name}: {e}")
                continue

            # データを追加
            all_data.append({
                'subject_id': user_id,
                'activity': activity_id,
                'data': sensor_data,
                'repetition': repetition,
            })

        logger.info(
            f"Loaded {len(all_data)} samples from {len(set(d['subject_id'] for d in all_data))} subjects"
        )

        return all_data

    def extract_features(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        特徴量を抽出

        手順:
        1. 各サンプルを32Hz → 30Hzにリサンプリング
        2. 各サンプルをスライディングウィンドウで分割
        3. ウィンドウごとに保存

        Args:
            raw_data: load_raw_data()の出力

        Returns:
            (features, labels, subjects) のタプル
                - features: (N, C, T) - N個のウィンドウ、C=3チャンネル、T=150サンプル
                - labels: (N,) - アクティビティラベル
                - subjects: (N,) - 被験者ID
        """
        window_size = self.window_size  # 150サンプル @ 30Hz = 5秒
        overlap = self.overlap  # 75サンプル = 50% オーバーラップ

        all_features = []
        all_labels = []
        all_subjects = []

        logger.info(
            f"Extracting features with window_size={window_size}, overlap={overlap}"
        )

        for sample in raw_data:
            subject_id = sample['subject_id']
            activity = sample['activity']
            data = sample['data']  # (N, 3)

            # リサンプリング: 32Hz → 30Hz
            # 正確な比率: 30/32 = 15/16
            data_resampled = resample_poly(
                data,
                up=15,
                down=16,
                axis=0,
                padtype='line'
            )  # (N', 3)

            # スライディングウィンドウで分割
            num_samples = len(data_resampled)
            stride = window_size - overlap

            # 最低限のウィンドウサイズを満たさない場合はスキップ
            if num_samples < window_size:
                continue

            # ウィンドウを抽出
            for start in range(0, num_samples - window_size + 1, stride):
                end = start + window_size
                window = data_resampled[start:end, :]  # (T, 3)

                # (T, 3) → (3, T) に転置
                window = window.T  # (3, T)

                all_features.append(window)
                all_labels.append(activity)
                all_subjects.append(subject_id)

        # NumPy配列に変換
        features = np.array(all_features, dtype=np.float32)  # (N, 3, 150)
        labels = np.array(all_labels, dtype=np.int64)  # (N,)
        subjects = np.array(all_subjects, dtype=np.int64)  # (N,)

        logger.info(
            f"Extracted {len(features)} windows from {len(raw_data)} samples"
        )
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Unique activities: {np.unique(labels)}")
        logger.info(f"Unique subjects: {np.unique(subjects)}")

        return features, labels, subjects

    def save_processed_data(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """
        処理済みデータを保存

        ディレクトリ構造:
            processed_data_path/
                USER{subject_id:05d}/
                    Wrist/
                        ACC/
                            X.npy  # (N, T)
                            Y.npy
                            Z.npy
                            labels.npy  # (N,)

        Args:
            data: (features, labels, subjects) のタプル
                - features: (N, C, T) - C=3 (X, Y, Z)
                - labels: (N,)
                - subjects: (N,)
        """
        features, labels, subjects = data
        save_dir = Path(self.processed_data_path) / self.get_dataset_name()
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving processed data to {save_dir}")

        # 被験者ごとにデータを分割して保存
        unique_subjects = np.unique(subjects)
        users_dict = {}

        for subject_id in unique_subjects:
            # 該当被験者のインデックス
            subject_mask = subjects == subject_id

            subject_features = features[subject_mask]  # (n, 3, T)
            subject_labels = labels[subject_mask]  # (n,)

            # 保存先ディレクトリ
            subject_dir = save_dir / f"USER{subject_id:05d}" / "Wrist" / "ACC"
            subject_dir.mkdir(parents=True, exist_ok=True)

            # X, Y, Z軸を個別に保存（float16で容量削減）
            np.save(
                subject_dir / "X.npy",
                subject_features[:, 0, :].astype(np.float16)
            )
            np.save(
                subject_dir / "Y.npy",
                subject_features[:, 1, :].astype(np.float16)
            )
            np.save(
                subject_dir / "Z.npy",
                subject_features[:, 2, :].astype(np.float16)
            )

            # ラベルを保存
            np.save(subject_dir / "labels.npy", subject_labels)

            logger.debug(
                f"Saved {len(subject_features)} windows for USER{subject_id:05d}"
            )

            # users辞書に追加（可視化ツール用）
            user_id_str = f"USER{subject_id:05d}"
            users_dict[user_id_str] = {
                "sensor_modalities": {
                    "Wrist/ACC": {
                        "X_shape": [len(subject_labels), self.window_size],
                        "Y_shape": [len(subject_labels), self.window_size],
                        "Z_shape": [len(subject_labels), self.window_size],
                        "labels_shape": [len(subject_labels)],
                        "num_windows": len(subject_labels),
                        "unique_labels": sorted([int(l) for l in np.unique(subject_labels)])
                    }
                }
            }

        # メタデータを保存
        import json
        metadata = {
            'dataset': self.get_dataset_name(),
            'num_subjects': len(unique_subjects),
            'num_windows': len(features),
            'num_activities': self.num_activities,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'sampling_rate': self.target_sampling_rate,
            'original_sampling_rate': self.original_sampling_rate,
            'scale_factor': self.scale_factor,
            'sensor_names': ['Wrist'],
            'sensor_positions': ['Wrist'],
            'modalities': ['ACC'],
            'channels_per_modality': 3,
            'users': users_dict,
        }

        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(
            f"Successfully saved data for {len(unique_subjects)} subjects "
            f"({len(features)} windows total)"
        )
