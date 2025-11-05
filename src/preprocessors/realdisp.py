"""
REALDISP Activity Recognition Dataset 前処理

REALDISP (REAListic sensor DISPlacement) データセット:
- 33種類の身体活動（ウォーミングアップ、クールダウン、フィットネス運動）
- 17人の被験者
- 9つのセンサー（全身装着）
- 3つのシナリオ（ideal-placement, self-placement, induced-displacement）
- サンプリングレート: 未定（データから推定）
- 各センサー: 3軸加速度、3軸ジャイロ、3軸磁気、4Dクォータニオン
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
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

# REALDISP dataset URL
REALDISP_URL = "https://archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip"


@register_preprocessor('realdisp')
class RealDispPreprocessor(BasePreprocessor):
    """
    REALDISPデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # REALDISP固有の設定
        self.num_activities = 33
        self.num_subjects = 17
        self.num_sensors = 9

        # センサー名とボディパートマッピング
        self.sensor_names = [
            'LeftCalf',      # S1
            'LeftThigh',     # S2
            'RightCalf',     # S3
            'RightThigh',    # S4
            'Back',          # S5
            'LeftLowerArm',  # S6
            'LeftUpperArm',  # S7
            'RightLowerArm', # S8
            'RightUpperArm'  # S9
        ]

        # 各センサーのチャンネル構成
        # ACC(3) + GYRO(3) + MAG(3) + QUAT(4) = 13 channels per sensor
        self.channels_per_sensor = 13

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'ACC': (0, 3),    # 3軸加速度
            'GYRO': (3, 6),   # 3軸ジャイロ
            'MAG': (6, 9),    # 3軸磁気
            'QUAT': (9, 13)   # 4Dクォータニオン
        }

        # サンプリングレート
        self.original_sampling_rate = DATASETS.get('REALDISP', {}).get('original_sampling_rate', 50)  # Hz (推定値)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（データ単位を確認後に設定）
        self.scale_factor = DATASETS.get('REALDISP', {}).get('scale_factor', None)

        # 処理するシナリオ（デフォルトは理想配置）
        self.scenarios = config.get('scenarios', ['ideal'])  # ['ideal', 'self', 'mutual']

    def get_dataset_name(self) -> str:
        return 'realdisp'

    def download_dataset(self) -> None:
        """
        REALDISPデータセットをUCI MLリポジトリからダウンロードして解凍
        """
        logger.info("=" * 80)
        logger.info("Downloading REALDISP dataset")
        logger.info("=" * 80)

        realdisp_raw_path = self.raw_data_path / self.dataset_name

        # 既にデータが存在するかチェック
        if check_dataset_exists(realdisp_raw_path, required_files=['*.log']):
            logger.warning(f"REALDISP data already exists at {realdisp_raw_path}")
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Skipping download")
                return

        try:
            # 1. ダウンロード
            logger.info("Step 1/2: Downloading archive (2.5 GB, may take a while)")
            realdisp_raw_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path = realdisp_raw_path.parent / 'realdisp.zip'
            download_file(REALDISP_URL, zip_path, desc='Downloading REALDISP')

            # 2. 解凍してデータ整理
            logger.info("Step 2/2: Extracting and organizing data")
            extract_to = realdisp_raw_path.parent / 'realdisp_temp'
            extract_archive(zip_path, extract_to, desc='Extracting REALDISP')
            self._organize_realdisp_data(extract_to, realdisp_raw_path)

            # クリーンアップ
            cleanup_temp_files(extract_to)
            if zip_path.exists():
                zip_path.unlink()

            logger.info("=" * 80)
            logger.info(f"SUCCESS: REALDISP dataset downloaded to {realdisp_raw_path}")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Failed to download REALDISP dataset: {e}", exc_info=True)
            raise

    def _organize_realdisp_data(self, extracted_path: Path, target_path: Path) -> None:
        """
        REALDISPデータを適切なディレクトリ構造に整理

        Args:
            extracted_path: 解凍されたデータのパス
            target_path: 整理後の保存先パス（data/raw/realdisp）
        """
        import shutil

        logger.info(f"Organizing REALDISP data from {extracted_path} to {target_path}")

        # 解凍されたデータのルートを見つける
        # ZIPファイルの構造を探索
        log_files = list(extracted_path.rglob('*.log'))

        if not log_files:
            raise FileNotFoundError(
                f"No .log files found in extracted archive at {extracted_path}"
            )

        logger.info(f"Found {len(log_files)} log files")

        # ターゲットディレクトリを作成
        target_path.mkdir(parents=True, exist_ok=True)

        # すべてのlogファイルをフラットな構造でコピー
        for log_file in log_files:
            dest_file = target_path / log_file.name
            shutil.copy2(log_file, dest_file)
            logger.info(f"  Copied: {log_file.name}")

        logger.info(f"Organized {len(log_files)} files to {target_path}")

    def _estimate_sampling_rate(self, timestamps: np.ndarray) -> float:
        """
        タイムスタンプから サンプリングレートを推定

        Args:
            timestamps: タイムスタンプ配列（秒単位）

        Returns:
            推定サンプリングレート（Hz）
        """
        # 時間差の中央値からサンプリングレートを推定
        time_diffs = np.diff(timestamps)
        median_diff = np.median(time_diffs)

        if median_diff > 0:
            sampling_rate = 1.0 / median_diff
            return sampling_rate
        else:
            logger.warning("Could not estimate sampling rate from timestamps")
            return 50.0  # デフォルト値

    def load_raw_data(self) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        REALDISPの生データを被験者ごと、シナリオごとに読み込む

        想定フォーマット:
        - data/raw/realdisp/*.log
        - ファイル名: subjectXX_scenario.log

        Returns:
            person_data: {person_id: {scenario: (data, labels)}} の辞書
                data: (num_samples, num_sensors * 13) の配列
                labels: (num_samples,) の配列（1-indexed、A1-A33）
        """
        raw_path = self.raw_data_path / self.dataset_name

        if not raw_path.exists():
            raise FileNotFoundError(
                f"REALDISP raw data not found at {raw_path}\n"
                "Expected structure: data/raw/realdisp/*.log\n"
                "Please download the dataset first using --download flag or manually."
            )

        # ログファイルを検索
        log_files = sorted(raw_path.glob('*.log'))

        if not log_files:
            raise FileNotFoundError(f"No .log files found in {raw_path}")

        logger.info(f"Found {len(log_files)} log files")

        # 被験者ごと、シナリオごとにデータを格納
        result = {}

        for log_file in log_files:
            # ファイル名から被験者IDとシナリオを抽出
            # 例: subject15_mutual5.log -> subject=15, scenario=mutual
            filename = log_file.stem  # 拡張子を除いたファイル名

            try:
                # ファイル名のパース
                parts = filename.split('_')
                if len(parts) < 2:
                    logger.warning(f"Unexpected filename format: {filename}")
                    continue

                # 被験者番号の抽出（"subject15" -> 15）
                subject_str = parts[0]
                if not subject_str.startswith('subject'):
                    continue

                person_id = int(subject_str.replace('subject', ''))

                # シナリオの抽出（"mutual5" -> "mutual", "ideal" -> "ideal"）
                scenario_str = parts[1]
                if scenario_str.startswith('mutual'):
                    scenario = 'mutual'
                elif scenario_str.startswith('self'):
                    scenario = 'self'
                elif scenario_str.startswith('ideal'):
                    scenario = 'ideal'
                else:
                    logger.warning(f"Unknown scenario in {filename}: {scenario_str}")
                    continue

                # 設定で指定されたシナリオのみ処理
                if scenario not in self.scenarios:
                    logger.info(f"Skipping {filename} (scenario '{scenario}' not in config)")
                    continue

                logger.info(f"Loading {filename} (Subject={person_id}, Scenario={scenario})")

                # データの読み込み
                # Column format:
                # 0: timestamp_sec
                # 1: timestamp_usec
                # 2-14: S1 [ACC(3), GYRO(3), MAG(3), QUAT(4)]
                # 15-27: S2 ...
                # ...
                # 106-118: S9
                # 119: label

                df = pd.read_csv(log_file, sep=r'\s+', header=None)

                if len(df.columns) != 120:
                    logger.warning(
                        f"Unexpected number of columns in {log_file}: "
                        f"{len(df.columns)} (expected 120)"
                    )
                    continue

                # タイムスタンプ（秒 + マイクロ秒）
                timestamps = df.iloc[:, 0].values + df.iloc[:, 1].values / 1e6

                # サンプリングレートの推定（最初のファイルのみ）
                if self.original_sampling_rate is None:
                    self.original_sampling_rate = self._estimate_sampling_rate(timestamps)
                    logger.info(f"Estimated sampling rate: {self.original_sampling_rate:.2f} Hz")

                # センサーデータ抽出（列2-118）
                sensor_data = df.iloc[:, 2:119].values.astype(np.float32)
                # sensor_data: (num_samples, 117) = 9 sensors * 13 channels

                # ラベル抽出（列119、1-indexed A1-A33 -> 0-indexed 0-32）
                labels = df.iloc[:, 119].values.astype(int) - 1

                # person_idごとに辞書を初期化
                if person_id not in result:
                    result[person_id] = {}

                result[person_id][scenario] = (sensor_data, labels)
                logger.info(
                    f"  USER{person_id:05d} ({scenario}): "
                    f"{sensor_data.shape}, Labels: {labels.shape}"
                )

            except Exception as e:
                logger.error(f"Error loading {log_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def clean_data(
        self,
        data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        データのクリーニングとリサンプリング

        Args:
            data: {person_id: {scenario: (data, labels)}} の辞書

        Returns:
            クリーニング・リサンプリング済み {person_id: {scenario: (data, labels)}}
        """
        cleaned = {}

        for person_id, scenario_data in data.items():
            cleaned[person_id] = {}

            for scenario, (person_data, labels) in scenario_data.items():
                # 無効なサンプルを除去
                cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)

                # リサンプリング
                if self.original_sampling_rate and self.original_sampling_rate != self.target_sampling_rate:
                    resampled_data, resampled_labels = resample_timeseries(
                        cleaned_data,
                        cleaned_labels,
                        self.original_sampling_rate,
                        self.target_sampling_rate
                    )
                    cleaned[person_id][scenario] = (resampled_data, resampled_labels)
                    logger.info(
                        f"USER{person_id:05d} ({scenario}) cleaned and resampled: "
                        f"{resampled_data.shape}"
                    )
                else:
                    cleaned[person_id][scenario] = (cleaned_data, cleaned_labels)
                    logger.info(
                        f"USER{person_id:05d} ({scenario}) cleaned: "
                        f"{cleaned_data.shape}"
                    )

        return cleaned

    def extract_features(
        self,
        data: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
    ) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化）

        Args:
            data: {person_id: {scenario: (data, labels)}} の辞書

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
            例: {'LeftCalf/ACC': {'X': (N, 3, 150), 'Y': (N,)}}
        """
        processed = {}

        for person_id, scenario_data in data.items():
            logger.info(f"Processing USER{person_id:05d}")

            processed[person_id] = {}

            # シナリオごとに処理（ideal, self, mutualを統合）
            # 各シナリオのデータを個別に処理
            for scenario, (person_data, labels) in scenario_data.items():
                logger.info(f"  Scenario: {scenario}")

                # 各センサーについて処理
                for sensor_idx, sensor_name in enumerate(self.sensor_names):
                    # センサーのチャンネル範囲を計算
                    sensor_start_ch = sensor_idx * self.channels_per_sensor
                    sensor_end_ch = sensor_start_ch + self.channels_per_sensor

                    # センサーのデータを抽出
                    sensor_data = person_data[:, sensor_start_ch:sensor_end_ch]

                    # スライディングウィンドウ適用
                    windowed_data, windowed_labels = create_sliding_windows(
                        sensor_data,
                        labels,
                        window_size=self.window_size,
                        stride=self.stride,
                        drop_last=False,
                        pad_last=True
                    )
                    # windowed_data: (num_windows, window_size, 13)

                    # 各モダリティに分割
                    for modality_name, (mod_start_ch, mod_end_ch) in self.sensor_modalities.items():
                        # モダリティのチャンネルを抽出
                        modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]
                        # modality_data: (num_windows, window_size, channels)

                        # スケーリング適用（加速度のみ）
                        if self.scale_factor is not None and modality_name == 'ACC':
                            modality_data = modality_data / self.scale_factor
                            logger.info(
                                f"    Applied scale_factor={self.scale_factor} to "
                                f"{sensor_name}/{modality_name}"
                            )

                        # 形状を変換: (num_windows, window_size, C) -> (num_windows, C, window_size)
                        modality_data = np.transpose(modality_data, (0, 2, 1))

                        # float16に変換
                        modality_data = modality_data.astype(np.float16)

                        # センサー/モダリティの階層構造
                        # シナリオ名を含めるかどうかは要検討
                        # とりあえずシナリオを含めずに統合
                        sensor_modality_key = f"{sensor_name}/{modality_name}"

                        # 既存のデータがあれば連結、なければ新規作成
                        if sensor_modality_key in processed[person_id]:
                            # 複数シナリオのデータを連結
                            existing_X = processed[person_id][sensor_modality_key]['X']
                            existing_Y = processed[person_id][sensor_modality_key]['Y']

                            processed[person_id][sensor_modality_key] = {
                                'X': np.concatenate([existing_X, modality_data], axis=0),
                                'Y': np.concatenate([existing_Y, windowed_labels], axis=0)
                            }
                        else:
                            processed[person_id][sensor_modality_key] = {
                                'X': modality_data,
                                'Y': windowed_labels
                            }

            # 統計情報の出力
            for sensor_modality_key, arrays in processed[person_id].items():
                X = arrays['X']
                Y = arrays['Y']
                logger.info(
                    f"  {sensor_modality_key}: X.shape={X.shape}, Y.shape={Y.shape}"
                )

        return processed

    def save_processed_data(self, data: Dict[int, Dict[str, Dict[str, np.ndarray]]]) -> None:
        """
        処理済みデータを保存

        Args:
            data: {person_id: {sensor_modality: {'X': data, 'Y': labels}}}

        保存形式:
            data/processed/realdisp/USER00001/LeftCalf/ACC/X.npy, Y.npy
            data/processed/realdisp/USER00001/LeftCalf/GYRO/X.npy, Y.npy
            data/processed/realdisp/USER00001/LeftCalf/MAG/X.npy, Y.npy
            data/processed/realdisp/USER00001/LeftCalf/QUAT/X.npy, Y.npy
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
            'scale_factor': self.scale_factor,
            'data_dtype': 'float16',
            'scenarios': self.scenarios,
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
                    f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}"
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
