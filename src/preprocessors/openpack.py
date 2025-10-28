"""
OpenPack データセット前処理

OpenPack データセット:
- **アクティビティ**: 11クラス（0-9 + undefined/-1）の物流作業操作
  - 0: Assemble, 1: Insert, 2: Put, 3: Walk, 4: Pick,
  - 5: Scan, 6: Press, 7: Open, 8: Close, 9: Other
  - -1: Undefined（無操作・ラベルなし）
- **被験者**: データセットに応じて可変（通常U0101-U0110など）
- **センサー**: 4つのATR TSND151 IMUセンサー（atr01, atr02, atr03, atr04）
  - 各センサー: ACC (3軸), GYRO (3軸), QUAT (4値) = 10チャンネル
  - 合計: 40チャンネル（4センサー × 10チャンネル）
- **サンプリングレート**: 30Hz
- **単位**:
  - ACC: G（重力加速度）- すでに正規化済み
  - GYRO: dps（degrees per second）
  - QUAT: 無次元（クォータニオン）

参考: https://open-pack.github.io/
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
    get_class_distribution
)
from .common import (
    download_file,
    extract_archive,
    cleanup_temp_files,
    check_dataset_exists
)
from . import register_preprocessor

logger = logging.getLogger(__name__)


# OpenPack データセットのURL
OPENPACK_URL = "https://zenodo.org/records/8145223/files/preprocessed-IMU-with-operation-labels.zip?download=1"


@register_preprocessor('openpack')
class OpenPackPreprocessor(BasePreprocessor):
    """
    OpenPackデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # OpenPack固有の設定
        self.num_activities = 10  # 操作クラス（0-9）+ undefined (-1)で実質11クラス
        self.num_subjects = None  # データセットに応じて可変（動的に検出）
        self.num_sensors = 4  # ATR TSND151 IMU × 4
        self.num_channels = 40  # 4 sensors × 10 channels (acc:3 + gyro:3 + quat:4)

        # センサー名とチャンネルマッピング
        self.sensor_names = ['atr01', 'atr02', 'atr03', 'atr04']

        # 各センサーのチャンネル数（ACC:3 + GYRO:3 + QUAT:4）
        self.channels_per_sensor = 10

        # センサーのチャンネル範囲（全体の中での位置）
        # 後でデータ読み込み時に動的に設定
        self.sensor_channel_ranges = {}

        # モダリティ（各センサー内のチャンネル分割）
        self.modalities = ['ACC', 'GYRO', 'QUAT']
        self.modality_channel_ranges = {
            'ACC': (0, 3),    # 3軸加速度
            'GYRO': (3, 6),   # 3軸ジャイロ
            'QUAT': (6, 10)   # 4値クォータニオン
        }

        # サンプリングレート
        self.original_sampling_rate = 30  # Hz (OpenPackのオリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

    def get_dataset_name(self) -> str:
        return 'openpack'

    def download_dataset(self) -> None:
        """
        OpenPackデータセットをダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading OpenPack dataset")
        logger.info("=" * 80)

        openpack_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(openpack_raw_path, required_files=['*.csv']):
            logger.warning(f"OpenPack data already exists at {openpack_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive (491 MB)")
            openpack_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = openpack_raw_path.parent / 'openpack.zip'
            download_file(OPENPACK_URL, zip_path, desc='Downloading OpenPack')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = openpack_raw_path.parent / 'openpack_temp'
            extract_archive(zip_path, extract_to, desc='Extracting OpenPack')
            self._organize_openpack_data(extract_to, openpack_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: OpenPack dataset downloaded to {openpack_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download OpenPack dataset: {e}", exc_info=True)
            raise

    def _organize_openpack_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        OpenPackデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/openpack）
        """
        import shutil
        from tqdm import tqdm

        logger.info(f"Organizing OpenPack data from {extracted_path} to {target_path}")

        # 解凍されたデータのルートを見つける
        data_root = extracted_path / "imuWithOperationLabel"

        if not data_root.exists():
            # 別の構造を試す
            possible_roots = list(extracted_path.rglob("imuWithOperationLabel"))
            if possible_roots:
                data_root = possible_roots[0]
            else:
                raise FileNotFoundError(f"Could not find imuWithOperationLabel directory in {extracted_path}")

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        # CSVファイルをコピー
        csv_files = list(data_root.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"Could not find CSV files in {data_root}")

        for csv_file in tqdm(csv_files, desc='Organizing files'):
            target_file = target_path / csv_file.name
            shutil.copy2(csv_file, target_file)

        logger.info(f"Data organized at: {target_path}")
        logger.info(f"Found {len(csv_files)} CSV files")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        OpenPackの生データを被験者ごとに読み込む

        想定フォーマット:
        - data/raw/openpack/U0101-S0100.csv
        - 各ファイル: unixtim, operation, atr01/acc_x, ..., atr04/quat_z

        Returns:
            person_data: {person_id: (data, labels)} の辞書
                data: (num_samples, 40) の配列
                labels: (num_samples,) の配列
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"OpenPack raw data not found at {raw_path}\n"
                "Expected structure: data/raw/openpack/U0101-S0100.csv"
            )

        # CSVファイルを取得
        csv_files = sorted(raw_path.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {raw_path}")

        # 被験者ごとにデータを格納
        person_data = {}

        for csv_file in csv_files:
            # ファイル名から被験者IDを抽出 (U0101-S0100.csv -> U0101)
            user_id_str = csv_file.stem.split('-')[0]  # "U0101"

            try:
                # U0101 -> 101 -> USER00001
                # OpenPackのユーザーIDは101-111, 201-210の範囲
                # 101-111 -> USER00001-USER00011 (11人)
                # 201-210 -> USER00012-USER00021 (10人)
                original_user_id = int(user_id_str[1:])

                if 101 <= original_user_id <= 111:
                    user_id = f"USER{(original_user_id - 100):05d}"  # 101 -> USER00001, 111 -> USER00011
                elif 201 <= original_user_id <= 210:
                    user_id = f"USER{(original_user_id - 189):05d}"  # 201 -> USER00012, 210 -> USER00021
                else:
                    logger.warning(f"Unexpected user ID: {user_id_str}, skipping")
                    continue

                # データ読み込み
                df = pd.read_csv(csv_file)

                # センサーデータのカラムを取得（unixtime, operationを除く）
                sensor_columns = [col for col in df.columns if '/' in col]

                # センサーデータとラベルを分離
                sensor_data = df[sensor_columns].values
                labels = df['operation'].values.astype(int)

                # operation 0（無操作）を-1に変換、それ以外を0-indexedに変換
                # 0 -> -1 (未定義クラス)
                # 1 -> 0, 2 -> 1, ... (有効なクラス)
                labels = np.where(labels == 0, -1, labels - 1)

                # 被験者ごとに結合
                if user_id not in person_data:
                    person_data[user_id] = {'data': [], 'labels': []}

                person_data[user_id]['data'].append(sensor_data)
                person_data[user_id]['labels'].append(labels)

                logger.debug(f"Loaded {csv_file.name}: {sensor_data.shape}")

            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
                continue

        # 各被験者のデータを結合
        result = {}
        for user_id, data_dict in person_data.items():
            if data_dict['data']:
                data = np.vstack(data_dict['data'])
                labels = np.hstack(data_dict['labels'])
                result[user_id] = (data, labels)
                logger.info(f"{user_id}: {data.shape}, Labels: {labels.shape}")

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
            # 無効なサンプルを除去
            cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)
            cleaned[person_id] = (cleaned_data, cleaned_labels)
            logger.info(f"{person_id} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化）

        Args:
            data: {person_id: (data, labels)} の辞書

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            例: {'atr01/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing {person_id}")

            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                # センサーのチャンネル範囲を計算
                sensor_start_ch = sensor_idx * self.channels_per_sensor
                sensor_end_ch = sensor_start_ch + self.channels_per_sensor

                # センサーのチャンネルを抽出
                sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]  # (samples, 10)

                # スライディングウィンドウ適用（最後のウィンドウはパディング）
                windowed_data, windowed_labels = create_sliding_windows(
                    sensor_data,
                    labels,
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_last=False,
                    pad_last=True  # OpenPack: 150に満たない場合はパディング
                )
                # windowed_data: (num_windows, window_size, 10)

                # 各モダリティに分割
                for modality_name in self.modalities:
                    mod_start_ch, mod_end_ch = self.modality_channel_ranges[modality_name]

                    # モダリティのチャンネルを抽出
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                    # modality_data: (num_windows, window_size, C)

                    # 形状を変換: (num_windows, window_size, C) -> (num_windows, C, window_size)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

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
            data/processed/openpack/USER00101/atr01/ACC/X.npy, Y.npy
            data/processed/openpack/USER00101/atr01/GYRO/X.npy, Y.npy
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
            'users': {}
        }

        for person_id, sensor_modality_data in data.items():
            # person_id is already in "USER00001" format
            user_path = base_path / person_id
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                # X.npy, Y.npy を保存（float16で効率化）
                X = arrays['X'].astype(np.float16)  # (num_windows, C, window_size)
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
                    f"Saved {person_id}/{sensor_modality_name}: "
                    f"X{X.shape}, Y{Y.shape}"
                )

            total_stats['users'][person_id] = user_stats

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
