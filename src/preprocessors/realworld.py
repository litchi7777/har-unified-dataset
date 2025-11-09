import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
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
        # 実データ上はすべて小文字(head, chest, ...)なので、ここをlower()して使う
        self.sensor_names = ['Chest', 'Forearm', 'Head', 'Shin', 'Thigh', 'UpperArm', 'Waist']

        # モダリティ（実データだと acc / gyr / mag / gps があるが、ここでは3つだけ扱う）
        self.modality_names = ['ACC', 'GYR', 'MAG']

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
        # RealWorldデータセットの活動名は小文字
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

    # ★ここを書き換えた★
    def _load_sensor_data(self, proband_path: Path, sensor: str, modality: str) -> Optional[Dict[str, List[pd.DataFrame]]]:
        """
        実際のRealWorld2016配布形式（activityごとのzipの中にsensorごとのcsvがある）に対応したローダ

        例:
            proband1/data/acc_climbingdown_csv.zip
                ├── acc_climbingdown_head.csv
                ├── acc_climbingdown_waist.csv
                └── ...
        という構造を想定する。

        Args:
            proband_path: 被験者のディレクトリパス (…/proband1)
            sensor: センサー名（例: 'Chest'）
            modality: モダリティ名（例: 'ACC'）

        Returns:
            activity名をキー、当該activityのDataFrameリストを値とする辞書
        """
        sensor_lower = sensor.lower()     # chest, head, ...
        modality_lower = modality.lower() # acc, gyr, mag

        data_dir = proband_path / 'data'
        if not data_dir.exists():
            logger.warning(f"Data dir not found: {data_dir}")
            return None

        # proband1/data 内の「acc_***_csv.zip」を全部見る
        pattern = f"{modality_lower}_*_csv.zip"
        zip_files = sorted(data_dir.glob(pattern))

        if not zip_files:
            logger.warning(f"No {modality_lower} zip files found in {data_dir}")
            return None

        activity_chunks: Dict[str, List[pd.DataFrame]] = {}

        for zip_path in zip_files:
            # zipファイル名から activity を取る: acc_climbingdown_csv.zip → climbingdown
            # ["acc", "climbingdown", "csv"]
            parts = zip_path.stem.split('_')
            activity = None
            if len(parts) >= 2:
                activity = parts[1]  # climbingdown, walking, ...

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # このzipに入っているcsvのうち、このセンサーのものだけ読む
                    # 例: acc_climbingdown_head.csv
                    target_suffix = f"_{sensor_lower}.csv"  # "_head.csv"
                    for member in zf.namelist():
                        name_lower = member.lower()
                        if not member.lower().endswith(".csv"):
                            continue
                        if not member.lower().endswith(target_suffix):
                            continue

                        if modality_lower == 'gyr':
                            if not ('gyro' in name_lower or 'gyroscope' in name_lower):
                                continue
                        elif modality_lower == 'acc':
                            if not ('acc' in name_lower or 'accelerometer' in name_lower):
                                continue

                        with zf.open(member) as fp:
                            df = pd.read_csv(fp)

                        # カラム名のゆらぎに対応
                        # RealWorldは多くが attr_time, attr_x, attr_y, attr_z だが
                        # 他にも timestamp, x, y, z がある
                        # ここではそのままdfを返し、上位でx,y,z列を拾う
                        if activity:
                            df["activity"] = activity
                        activity_chunks.setdefault(activity or "unknown", []).append(df)

            except Exception as e:
                logger.warning(f"Error reading zip {zip_path}: {e}")
                continue

        if not activity_chunks:
            logger.warning(
                f"No CSV found for sensor={sensor_lower}, modality={modality_lower} in {data_dir}"
            )
            return None

        combined_chunks: Dict[str, List[pd.DataFrame]] = {}
        for activity_name, dfs in activity_chunks.items():
            combined_chunks[activity_name] = [pd.concat(dfs, ignore_index=True)]

        return combined_chunks

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

        fuse_sources = ('ACC', 'GYR')
        result = {}

        # proband1～proband15を読み込み
        for person_id in range(1, self.num_subjects + 1):
            proband_path = dataset_root / f'proband{person_id}'

            if not proband_path.exists():
                logger.warning(f"Proband {person_id} not found at {proband_path}")
                continue

            logger.info(f"Loading USER{person_id:05d} from {proband_path.name}")

            person_entry: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

            # 各センサーについて処理
            for sensor in self.sensor_names:
                sensor_data_store: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                modality_chunks: Dict[str, Dict[str, List[pd.DataFrame]]] = {}

                for modality in self.modality_names:
                    activity_chunks = self._load_sensor_data(proband_path, sensor, modality)

                    if not activity_chunks:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}/{modality}: No data loaded"
                        )
                        continue

                    modality_chunks[modality] = activity_chunks

                    # 磁気など単独モダリティは従来通り保存
                    if modality not in fuse_sources:
                        combined_df = pd.concat(
                            [fragment for fragments in activity_chunks.values() for fragment in fragments],
                            ignore_index=True
                        )
                        extracted = self._extract_xyz_data_with_labels(combined_df)
                        if extracted is None:
                            continue
                        sensor_data, labels = extracted
                        sensor_data_store[modality] = (sensor_data, labels)
                        logger.info(
                            f"  {sensor}/{modality}: {sensor_data.shape}, "
                            f"Labels: {np.unique(labels)}"
                        )

                # ACC/GYR の同期データ
                if all(src in modality_chunks for src in fuse_sources):
                    fused = self._build_synchronized_activity_chunks(
                        modality_chunks[fuse_sources[0]],
                        modality_chunks[fuse_sources[1]],
                        sensor
                    )
                    if fused is not None:
                        fused_data, fused_labels = fused
                        acc_data = fused_data[:, :3]
                        gyro_data = fused_data[:, 3:]
                        sensor_data_store['ACC'] = (acc_data, fused_labels)
                        sensor_data_store['GYR'] = (gyro_data, fused_labels)
                        logger.info(
                            f"  {sensor}/ACC(sync): {acc_data.shape}, Labels: {np.unique(fused_labels)}"
                        )
                        logger.info(
                            f"  {sensor}/GYR(sync): {gyro_data.shape}, Labels: {np.unique(fused_labels)}"
                        )
                else:
                    missing = [src for src in fuse_sources if src not in modality_chunks]
                    if missing:
                        logger.warning(
                            f"USER{person_id:05d}/{sensor}: missing modalities for fusion: {missing}"
                        )

                if sensor_data_store:
                    person_entry[sensor] = sensor_data_store

            if person_entry:
                result[person_id] = person_entry

        if not result:
            raise ValueError("No data loaded. Please check the raw data directory structure.")

        logger.info(f"Total users loaded: {len(result)}")
        return result

    def _extract_xyz_data_with_labels(
        self,
        df: pd.DataFrame
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        xyz_columns = self._get_xyz_columns(df)

        if len(xyz_columns) < 3:
            logger.warning("Could not find three axis columns in DataFrame")
            return None

        sensor_data = df[xyz_columns[:3]].values.astype(np.float32)

        if 'activity' in df.columns:
            activity_series = df['activity'].astype(str).str.lower()
            labels = activity_series.map(self.activity_mapping).fillna(-1).astype(int).values
        else:
            labels = np.full(len(sensor_data), -1, dtype=int)

        return sensor_data, labels

    def _get_xyz_columns(self, df: pd.DataFrame) -> List[str]:
        axis_candidates = {
            'x': ['attr_x', 'x'],
            'y': ['attr_y', 'y'],
            'z': ['attr_z', 'z']
        }

        xyz_columns: List[str] = []
        for axis in ('x', 'y', 'z'):
            candidates = axis_candidates[axis]
            column = next(
                (col for col in df.columns if col.lower() in candidates),
                None
            )
            if column:
                xyz_columns.append(column)

        return xyz_columns

    def _get_time_column(self, df: pd.DataFrame) -> Optional[str]:
        time_candidates = ['attr_time', 'timestamp', 'time', 'ts', 't']
        return next((col for col in df.columns if col.lower() in time_candidates), None)

    def _prepare_chunk_arrays(
        self,
        df: pd.DataFrame,
        activity_name: Optional[str]
    ) -> Optional[Tuple[Optional[np.ndarray], np.ndarray, int]]:
        xyz_columns = self._get_xyz_columns(df)
        if len(xyz_columns) < 3:
            logger.warning("Chunk is missing xyz columns")
            return None

        values = df[xyz_columns[:3]].values.astype(np.float32)
        time_col = self._get_time_column(df)
        timestamps = df[time_col].to_numpy(np.float64) if time_col else None

        candidate_activity = activity_name
        if not candidate_activity and 'activity' in df.columns and len(df['activity']):
            candidate_activity = str(df['activity'].iloc[0])

        label_id = self.activity_mapping.get(str(candidate_activity).lower(), -1) if candidate_activity else -1

        return timestamps, values, label_id

    def _align_chunk_arrays(
        self,
        acc_time: Optional[np.ndarray],
        acc_values: np.ndarray,
        gyro_time: Optional[np.ndarray],
        gyro_values: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if acc_values.size == 0 or gyro_values.size == 0:
            return None

        if acc_time is not None and gyro_time is not None:
            start = max(acc_time[0], gyro_time[0])
            end = min(acc_time[-1], gyro_time[-1])
            if end <= start:
                return None

            acc_mask = (acc_time >= start) & (acc_time <= end)
            gyro_mask = (gyro_time >= start) & (gyro_time <= end)

            acc_values = acc_values[acc_mask]
            gyro_values = gyro_values[gyro_mask]

        min_len = min(len(acc_values), len(gyro_values))
        if min_len == 0:
            return None

        return acc_values[:min_len], gyro_values[:min_len]

    def _build_synchronized_activity_chunks(
        self,
        acc_chunks: Dict[str, List[pd.DataFrame]],
        gyro_chunks: Dict[str, List[pd.DataFrame]],
        sensor_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        shared_activities = sorted(set(acc_chunks.keys()) & set(gyro_chunks.keys()))
        if not shared_activities:
            logger.warning(f"{sensor_name}: no overlapping activities for ACC/GYR fusion")
            return None

        fused_segments: List[np.ndarray] = []
        fused_labels: List[np.ndarray] = []

        for activity in shared_activities:
            acc_list = acc_chunks.get(activity, [])
            gyro_list = gyro_chunks.get(activity, [])

            if not acc_list or not gyro_list:
                continue

            if len(acc_list) != len(gyro_list):
                logger.warning(
                    f"{sensor_name}/{activity}: chunk count mismatch (ACC={len(acc_list)}, GYR={len(gyro_list)})"
                )

            pair_count = min(len(acc_list), len(gyro_list))

            for idx in range(pair_count):
                acc_chunk = acc_list[idx]
                gyro_chunk = gyro_list[idx]

                acc_prepared = self._prepare_chunk_arrays(acc_chunk, activity)
                gyro_prepared = self._prepare_chunk_arrays(gyro_chunk, activity)

                if acc_prepared is None or gyro_prepared is None:
                    continue

                acc_time, acc_values, label_id = acc_prepared
                gyro_time, gyro_values, _ = gyro_prepared

                aligned = self._align_chunk_arrays(acc_time, acc_values, gyro_time, gyro_values)
                if aligned is None:
                    logger.warning(f"{sensor_name}/{activity}: failed to align ACC/GYR chunk {idx}")
                    continue

                acc_aligned, gyro_aligned = aligned
                fused_values = np.concatenate([acc_aligned, gyro_aligned], axis=1)
                labels = np.full(len(fused_values), label_id, dtype=int)

                fused_segments.append(fused_values.astype(np.float32))
                fused_labels.append(labels)

        if not fused_segments:
            return None

        fused_data = np.concatenate(fused_segments, axis=0)
        fused_label_array = np.concatenate(fused_labels, axis=0)
        return fused_data, fused_label_array

    def clean_data(
        self,
        data: Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]
    ) -> Dict[int, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
        """
        データのクリーニングとリサンプリング
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
