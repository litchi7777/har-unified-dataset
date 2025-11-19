"""
CAPTURE-24 (Large-scale daily living HAR) データセット前処理

CAPTURE-24 データセット:
- 151人の参加者が約24時間装着
- Axivity AX3手首装着型加速度計
- サンプリングレート: 100Hz (3軸加速度)
- 200以上の日常生活行動（Compendium of Physical Activity）
- ラベルスキーマ: Walmsley2020 or WillettsSpecific2018
- 参照: https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import zipfile
import urllib.request as urllib
from tqdm.auto import tqdm

from .base import BasePreprocessor
from .utils import (
    create_sliding_windows,
    filter_invalid_samples,
    resample_timeseries,
    get_class_distribution
)
from .common import (
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor
from ..dataset_info import DATASETS

logger = logging.getLogger(__name__)


# CAPTURE-24 データセットのURL
CAPTURE24_URL = "https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001" + \
                "/download_file?file_format=&safe_filename=capture24.zip&type_of_work=Dataset"

# データファイルとアノテーションファイルのパターン
DATAFILES_PATTERN = 'P[0-9][0-9][0-9].csv.gz'  # P001.csv.gz ... P151.csv.gz
ANNOFILE = 'annotation-label-dictionary.csv'


@register_preprocessor('capture24')
class Capture24Preprocessor(BasePreprocessor):
    """
    CAPTURE-24データセット用の前処理クラス

    Features:
    - 151 participants, ~4000 hours total
    - Wrist-worn accelerometer (100Hz, 3-axis)
    - 200+ activities from Compendium of Physical Activity
    - Labels: Walmsley2020 or WillettsSpecific2018 schema
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # CAPTURE-24固有の設定
        self.num_participants = 151
        self.sensor_names = ['Wrist']  # 手首装着のみ

        # ラベルスキーマの選択
        self.label_schema = config.get('label_schema', 'Walmsley2020')  # or 'WillettsSpecific2018'
        logger.info(f"Using label schema: {self.label_schema}")

        # サブセット化（オプション: メモリ/時間節約のため）
        self.max_participants = config.get('max_participants', None)  # None = all 151
        if self.max_participants:
            logger.info(f"Limiting to first {self.max_participants} participants")

        # モダリティ
        self.sensor_modalities = {
            'Wrist': {
                'ACC': (0, 3),  # 3軸加速度 (x, y, z)
            }
        }

        # サンプリングレート
        self.original_sampling_rate = 100  # Hz (Axivity AX3)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz (80% overlap)

        # スケーリング係数（データセット確認後に設定）
        self.scale_factor = DATASETS.get('CAPTURE24', {}).get('scale_factor', None)

        # ラベルマッピング（動的に読み込み）
        self.label_to_id = {}
        self.id_to_label = {}
        self.anno_df = None  # annotation-label-dictionary.csv

    def get_dataset_name(self) -> str:
        return 'capture24'

    def download_dataset(self) -> None:
        """
        CAPTURE-24データセットをダウンロードして解凍

        Warning: 6.5GB以上の大容量ファイル
        """
        zip_path = self.raw_data_path / 'capture24.zip'
        extract_dir = self.raw_data_path / 'capture24'

        # ダウンロード済みかチェック
        if extract_dir.exists():
            csv_files = list(extract_dir.glob(DATAFILES_PATTERN))
            anno_file = extract_dir / ANNOFILE

            if len(csv_files) >= 151 and anno_file.exists():
                logger.info(f"CAPTURE-24 data already exists at {extract_dir}")
                return

        # ダウンロード
        if not zip_path.exists():
            logger.info(f"Downloading CAPTURE-24 dataset (6.5GB+)...")
            logger.info(f"URL: {CAPTURE24_URL}")
            logger.info("This may take 10-30 minutes depending on your connection.")

            with tqdm(total=6.9e9, unit="B", unit_scale=True, unit_divisor=1024,
                      miniters=1, ascii=True, desc="Downloading capture24.zip") as pbar:
                urllib.urlretrieve(
                    CAPTURE24_URL,
                    filename=str(zip_path),
                    reporthook=lambda b, bsize, tsize: pbar.update(bsize)
                )

            logger.info(f"Download completed: {zip_path}")
        else:
            logger.info(f"Using existing zip file: {zip_path}")

        # 解凍
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting CAPTURE-24 data...")
        with zipfile.ZipFile(zip_path, "r") as f:
            for member in tqdm(f.namelist(), desc="Unzipping"):
                try:
                    f.extract(member, self.raw_data_path)
                except zipfile.error as e:
                    logger.warning(f"Error extracting {member}: {e}")

        logger.info(f"Extraction completed: {extract_dir}")

    def load_annotation_dictionary(self) -> pd.DataFrame:
        """
        annotation-label-dictionary.csvを読み込み

        Returns:
            DataFrame with columns: annotation, label:Walmsley2020, label:WillettsSpecific2018, etc.
        """
        anno_file = self.raw_data_path / 'capture24' / ANNOFILE

        if not anno_file.exists():
            raise FileNotFoundError(
                f"Annotation dictionary not found: {anno_file}\n"
                "Please run download_dataset() first."
            )

        # CSVを読み込み（annotationをindex）
        anno_df = pd.read_csv(anno_file, index_col='annotation', dtype=str)
        logger.info(f"Loaded annotation dictionary: {anno_file}")
        logger.info(f"  Available schemas: {[c for c in anno_df.columns if c.startswith('label:')]}")

        return anno_df

    def build_label_mapping(self, anno_df: pd.DataFrame) -> None:
        """
        選択したラベルスキーマから label_to_id と id_to_label を構築

        Args:
            anno_df: annotation-label-dictionary DataFrame
        """
        schema_col = f'label:{self.label_schema}'

        if schema_col not in anno_df.columns:
            raise ValueError(
                f"Schema '{self.label_schema}' not found in annotation dictionary.\n"
                f"Available: {anno_df.columns.tolist()}"
            )

        # ユニークなラベルを取得（NAを除く）
        unique_labels = anno_df[schema_col].dropna().unique()
        unique_labels = sorted(unique_labels)

        # label -> id マッピング
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_to_id['NA'] = -1  # 未定義クラス

        # id -> label マッピング
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        logger.info(f"Built label mapping for schema '{self.label_schema}':")
        logger.info(f"  Number of classes: {len(unique_labels)}")
        logger.info(f"  Labels: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")

    def load_participant_data(self, participant_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        1人の参加者のデータを読み込み

        Args:
            participant_file: P###.csv.gz ファイル

        Returns:
            data: (num_samples, 3) - x, y, z 加速度
            labels: (num_samples,) - ラベルID
        """
        # CSVを読み込み
        df = pd.read_csv(
            participant_file,
            index_col='time',
            parse_dates=['time'],
            dtype={'x': 'f4', 'y': 'f4', 'z': 'f4', 'annotation': 'string'}
        )

        # 加速度データ
        acc_data = df[['x', 'y', 'z']].to_numpy()

        # アノテーションをラベルIDに変換
        annotations = df['annotation'].to_numpy()
        schema_col = f'label:{self.label_schema}'

        labels = np.full(len(annotations), -1, dtype=np.int32)
        for i, anno in enumerate(annotations):
            if pd.isna(anno):
                labels[i] = -1
            elif anno in self.anno_df.index:
                label_str = self.anno_df.loc[anno, schema_col]
                if pd.isna(label_str):
                    labels[i] = -1
                else:
                    labels[i] = self.label_to_id.get(label_str, -1)
            else:
                labels[i] = -1

        return acc_data, labels

    def load_raw_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        全参加者のデータを読み込み

        Returns:
            {participant_id: (data, labels)} の辞書
                data: (num_samples, 3) - 加速度
                labels: (num_samples,) - ラベルID
        """
        capture24_dir = self.raw_data_path / 'capture24'

        # アノテーション辞書を読み込み
        self.anno_df = self.load_annotation_dictionary()
        self.build_label_mapping(self.anno_df)

        # 参加者ファイルを取得
        participant_files = sorted(capture24_dir.glob(DATAFILES_PATTERN))

        if len(participant_files) == 0:
            raise FileNotFoundError(
                f"No participant files found in {capture24_dir}\n"
                "Please run download_dataset() first."
            )

        # サブセット化
        if self.max_participants:
            participant_files = participant_files[:self.max_participants]

        logger.info(f"Loading {len(participant_files)} participants...")

        all_data = {}
        for pfile in tqdm(participant_files, desc="Loading participants"):
            # 参加者ID (P001 -> 1)
            pid = int(pfile.stem.split('.')[0][1:])  # P001.csv.gz -> 1

            try:
                data, labels = self.load_participant_data(pfile)
                all_data[pid] = (data, labels)
                logger.info(f"P{pid:03d}: data={data.shape}, labels={labels.shape}, "
                           f"unique_labels={np.unique(labels)}")
            except Exception as e:
                logger.error(f"Error loading P{pid:03d}: {e}")

        logger.info(f"Loaded {len(all_data)} participants successfully")
        return all_data

    def clean_data(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニングとリサンプリング
        """
        cleaned = {}

        for pid, (person_data, labels) in data.items():
            # NaN除去
            valid_mask = ~np.isnan(person_data).any(axis=1)
            cleaned_data = person_data[valid_mask]
            cleaned_labels = labels[valid_mask]

            # リサンプリング (100Hz -> 30Hz)
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[pid] = (resampled_data, resampled_labels)
                logger.info(f"P{pid:03d} cleaned and resampled: {resampled_data.shape}")
            else:
                cleaned[pid] = (cleaned_data, cleaned_labels)
                logger.info(f"P{pid:03d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Returns:
            {participant_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for pid, (person_data, labels) in data.items():
            logger.info(f"Processing P{pid:03d}")
            processed[pid] = {}

            # Wristセンサーのみ
            sensor_name = 'Wrist'
            sensor_data = person_data  # すでに (N, 3)

            # スライディングウィンドウ適用
            windowed_data, windowed_labels = create_sliding_windows(
                sensor_data,
                labels,
                window_size=self.window_size,
                stride=self.stride,
                drop_last=False,
                pad_last=True
            )

            # ACCモダリティ
            modality_name = 'ACC'
            modality_data = windowed_data  # (num_windows, window_size, 3)

            # スケーリング適用（ACCのみ、scale_factorが定義されている場合）
            if self.scale_factor is not None:
                modality_data = modality_data / self.scale_factor
                logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

            # 形状変換: (N, T, C) -> (N, C, T)
            modality_data = np.transpose(modality_data, (0, 2, 1))

            # float16に変換（メモリ効率化）
            modality_data = modality_data.astype(np.float16)

            sensor_modality_key = f"{sensor_name}/{modality_name}"
            processed[pid][sensor_modality_key] = {
                'X': modality_data,
                'Y': windowed_labels
            }

            logger.info(f"  {sensor_modality_key}: X.shape={modality_data.shape}, "
                       f"Y.shape={windowed_labels.shape}")

        return processed

    def save_processed_data(
        self,
        data: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> None:
        """
        処理済みデータを保存

        保存形式:
            data/processed/capture24/P001/Wrist/ACC/X.npy, Y.npy
        """
        import json

        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        total_stats = {
            'dataset': self.dataset_name,
            'num_participants': len(data),
            'label_schema': self.label_schema,
            'num_classes': len(self.label_to_id) - 1,  # -1はNAを除く
            'sensor_names': self.sensor_names,
            'original_sampling_rate': self.original_sampling_rate,
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'normalization': 'none',
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'labels': self.id_to_label,
            'participants': {}
        }

        for pid, sensor_modality_data in data.items():
            participant_name = f"P{pid:03d}"
            participant_path = base_path / participant_name
            participant_path.mkdir(parents=True, exist_ok=True)

            participant_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = participant_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                X = arrays['X']
                Y = arrays['Y']

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                participant_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {participant_name}/{sensor_modality_name}: "
                           f"X{X.shape}, Y{Y.shape}")

            total_stats['participants'][participant_name] = participant_stats

        # メタデータ保存（NumPy型をJSON互換に変換）
        metadata_path = base_path / 'metadata.json'

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

        with open(metadata_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")
        logger.info(f"Preprocessing completed: {base_path}")
