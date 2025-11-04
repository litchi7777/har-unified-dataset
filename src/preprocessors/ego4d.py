"""
Ego4D IMU データ前処理

adjust_fps_Ego4d.py のロジックを本リポジトリの前処理フレームワークに適合させた実装。

主な処理:
- CSV を読み込み、タイムスタンプギャップや逆行で分割
- 連続区間ごとに実効サンプリング周波数を推定
- 高レートの場合は等間隔再配置→LPF付き polyphase で目標レート（既定30Hz）へ変換
- それ以外は目標レート等間隔へ線形補間
- 1秒（目標レート分のサンプル数）非オーバーラップ窓に分割
- 出力はセンサー "Head" の `ACC`/`GYRO` に分け、
  data/processed/ego4d/USERxxxxx/Head/{ACC|GYRO}/X.npy, Y.npy として保存

備考:
- Ego4D IMU にはアクティビティラベルが無いため、Y.npy にはウィンドウ開始時刻（秒）を保存する
  （他データセットのラベルと互換ではない点に注意）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging

import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.signal import resample_poly
import subprocess
import sys
import shutil

from .base import BasePreprocessor
from . import register_preprocessor


logger = logging.getLogger(__name__)


# 入力CSVの期待ヘッダ
REQUIRED_HEADER = [
    "component_idx",
    "component_timestamp_ms",
    "canonical_timestamp_ms",
    "gyro_x", "gyro_y", "gyro_z", "accl_x", "accl_y", "accl_z",
]

COLS_IMU = ["gyro_x", "gyro_y", "gyro_z", "accl_x", "accl_y", "accl_z"]
COL_COMP = "component_timestamp_ms"


def _detect_unit_ms(t_ms: np.ndarray) -> bool:
    t = np.asarray(t_ms, dtype=np.float64)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return False
    med = float(np.median(dt))
    return 0.5 <= med < 1000.0


def _separate_by_time_gap(df: pd.DataFrame, max_gap_sec: float) -> List[pd.DataFrame]:
    t = df[COL_COMP].to_numpy(np.float64)
    order = np.argsort(t)
    df = df.iloc[order].reset_index(drop=True)
    t = t[order]

    t_sec = (t - t[0]) / 1000.0
    dt = np.diff(t_sec)
    split_mask = (np.abs(dt) > max_gap_sec) | (dt < 0.0)
    split_idx = np.where(split_mask)[0] + 1

    parts = []
    s = 0
    for e in split_idx:
        parts.append(df.iloc[s:e].copy())
        s = e
    if s < len(df):
        parts.append(df.iloc[s:].copy())
    return parts


def _robust_fs(t_sec: np.ndarray) -> float:
    dt = np.diff(t_sec)
    dt = dt[dt > 0]
    if dt.size == 0:
        return np.nan
    q1, q3 = np.quantile(dt, [0.05, 0.95])
    dt_clip = dt[(dt >= q1) & (dt <= q3)]
    if dt_clip.size < 10:
        dt_clip = dt
    med = float(np.median(dt_clip))
    return 1.0 / med if med > 0 else np.nan


def _to_uniform_grid_at_fs(t_sec: np.ndarray, y_cols: List[np.ndarray], fs_est: float):
    dt_src = 1.0 / fs_est
    t0 = np.ceil(t_sec[0] / dt_src) * dt_src
    t1 = np.floor(t_sec[-1] / dt_src) * dt_src
    if t1 <= t0:
        return np.array([]), []
    n = int(round((t1 - t0) / dt_src)) + 1
    t_uni = t0 + np.arange(n, dtype=np.float64) * dt_src
    y_uni = [np.interp(t_uni, t_sec, y) for y in y_cols]
    return t_uni, y_uni


def _lpf_downsample_to_target(y_uni_cols: List[np.ndarray], fs_est: float, target_hz: float):
    ratio = Fraction.from_float(target_hz / fs_est).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    y_out_cols = [resample_poly(y, up, down) for y in y_uni_cols]
    return y_out_cols


@register_preprocessor('ego4d')
class Ego4dPreprocessor(BasePreprocessor):
    """
    Ego4D IMU 前処理

    設定 (config) で利用可能なキー例:
        - raw_data_path: 生データ基底 (既定: data/raw)
        - processed_data_path: 出力基底 (既定: data/processed)
        - ego4d_src_root: CSV探索ルート（未指定時は raw_data_path/ego4d）
        - target_sampling_rate: 既定 30
        - window_size: 既定 30 (samples)
        - stride: 既定 30 (非オーバーラップ)
        - max_gap_sec: 既定 1.0
        - min_dataset_span_sec: 既定 59/60
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.target_sampling_rate = float(config.get('target_sampling_rate', 30.0))
        self.window_size = int(config.get('window_size', int(self.target_sampling_rate)))
        self.stride = int(config.get('stride', self.window_size))
        self.max_gap_sec = float(config.get('max_gap_sec', 1.0))
        self.min_dataset_span_sec = float(config.get('min_dataset_span_sec', 59.0/60.0))

        # センサーとモダリティの想定（単一デバイス6ch: GYRO(3) + ACC(3)）
        self.sensor_name = 'Head'
        self.modalities = {
            'GYRO': (0, 3),
            'ACC': (3, 6),
        }

        # 入力ルート（未指定なら data/raw/ego4d 配下を想定）
        self.src_root = Path(config.get('ego4d_src_root', self.raw_data_path / self.dataset_name))

    def get_dataset_name(self) -> str:
        return 'ego4d'

    def download_dataset(self) -> None:
        """
        ego4d パッケージをインストールし、CLI で IMU データをダウンロード。

        実行コマンド（優先順）:
          1) ego4d --output_directory="<raw/ego4d>" --dataset=imu
          2) python -m ego4d --output_directory=... --dataset=imu

        既に CSV が存在する場合はスキップ。
        """
        output_dir = self.raw_data_path / self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # 既存チェック（CSVがあるならスキップ）
        existing_csv = list(output_dir.rglob('*.csv'))
        if existing_csv:
            logger.info(f"Ego4D IMU already present under {output_dir} (csv count={len(existing_csv)}), skipping download")
            return

        logger.info("Preparing to download Ego4D IMU via pip+CLI ...")

        # ego4d パッケージが無ければインストール
        try:
            import importlib.util as _importlib_util  # type: ignore
            if _importlib_util.find_spec('ego4d') is None:
                logger.info("Installing 'ego4d' package via pip ...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'ego4d', '--quiet'], check=True)
            else:
                logger.info("'ego4d' package already installed")
        except Exception as e:
            logger.error(f"Failed to ensure ego4d package: {e}")
            raise

        # CLI 実行
        cli_cmd = shutil.which('ego4d')
        cmd: List[str]
        if cli_cmd:
            cmd = [cli_cmd, f"--output_directory={str(output_dir)}", "--dataset=imu"]
        else:
            # console_script が見つからない場合は -m 経由を試す
            cmd = [sys.executable, '-m', 'ego4d', f"--output_directory={str(output_dir)}", "--dataset=imu"]

        logger.info(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"ego4d CLI failed with exit code {e.returncode}")
            logger.error(
                "\n\n===============\n\n"
                "Ego4D download failed (HTTP 403 from S3).\n\n"
                "You must request a license and obtain temporary AWS credentials before downloading Ego4D data. "
                "Please follow the official “Start Here” guide: https://ego4d-data.org/docs/start-here/\n\n"
                "==============="
            )
            raise
        except Exception as e:
            logger.error(f"Failed to run ego4d CLI: {e}")
            logger.error(
                "\n\n===============\n\n"
                "Ego4D download failed (HTTP 403 from S3).\n\n"
                "You must request a license and obtain temporary AWS credentials before downloading Ego4D data. "
                "Please follow the official “Start Here” guide: https://ego4d-data.org/docs/start-here/\n\n"
                "==============="
            )
            raise

        # 成功検証
        downloaded_csv = list(output_dir.rglob('*.csv'))
        if not downloaded_csv:
            logger.warning(f"Ego4D CLI finished but no CSV found under {output_dir}. Check CLI output structure.")
        else:
            logger.info(f"Ego4D IMU downloaded: {len(downloaded_csv)} CSV files under {output_dir}")

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        CSV を走査して、各ファイルを 1 ユーザー（擬似ID）扱いで
        (windows, window_start_time) を返す。

        Returns:
            {person_id: (windows, window_start_sec)}
            - windows: (N, window_size, 6)
            - window_start_sec: (N,)
        """
        if not self.src_root.exists():
            raise FileNotFoundError(f"Ego4D source root not found: {self.src_root}")

        csv_files = sorted(self.src_root.rglob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files under {self.src_root}")

        logger.info(f"Found {len(csv_files)} CSV files under {self.src_root}")

        result: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        person_id = 1

        for csv_path in csv_files:
            try:
                windows, t0 = self._process_one_csv(csv_path)
            except Exception as e:
                logger.warning(f"Failed to process {csv_path.name}: {e}")
                continue

            if windows.size == 0:
                logger.info(f"No valid windows for {csv_path.name}")
                continue

            result[person_id] = (windows, t0)
            logger.info(f"USER{person_id:05d}: windows={windows.shape[0]} from {csv_path.name}")
            person_id += 1

        if not result:
            raise ValueError("No windows generated from Ego4D CSVs")

        return result

    def _process_one_csv(self, in_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        # ヘッダ検証（1行目のみ簡易チェック）
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                header = [h.strip() for h in f.readline().strip().split(",")]
            if header != REQUIRED_HEADER:
                logger.info(f"[SKIP] {in_path.name}: header mismatch")
                return np.empty((0, self.window_size, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
        except Exception as e:
            logger.info(f"[SKIP] {in_path.name}: header read failed ({e})")
            return np.empty((0, self.window_size, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

        df = pd.read_csv(in_path)

        if COL_COMP not in df.columns or not df[COL_COMP].notna().any():
            return np.empty((0, self.window_size, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
        if not _detect_unit_ms(df[COL_COMP].to_numpy()):
            return np.empty((0, self.window_size, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

        parts = _separate_by_time_gap(df, self.max_gap_sec)

        windows: List[np.ndarray] = []
        t0_list: List[np.ndarray] = []

        for part in parts:
            if len(part) < 2:
                continue

            t_ms = part[COL_COMP].to_numpy(np.float64)
            part = part.loc[part[COL_COMP].notna()].reset_index(drop=True)
            t_ms = part[COL_COMP].to_numpy(np.float64)

            # 重複タイムスタンプの平均化
            t_unique, inv = np.unique(t_ms, return_inverse=True)
            if len(t_unique) != len(t_ms):
                agg = {c: "mean" for c in COLS_IMU}
                part["__grp__"] = inv
                part = part.groupby("__grp__", as_index=False).agg(agg)
                part = part.drop(columns=["__grp__"])
                t_ms = t_unique

            t_sec = (t_ms - t_ms[0]) / 1000.0
            if t_sec[-1] - t_sec[0] < self.min_dataset_span_sec:
                continue

            fs_est = _robust_fs(t_sec)
            if not np.isfinite(fs_est) or fs_est <= 0:
                continue

            y_cols_raw = [part[c].to_numpy(np.float64) for c in COLS_IMU]

            if fs_est > (self.target_sampling_rate + 15.0):
                # 高レート: 等間隔化→LPFダウンサンプル
                t_uni, y_uni_cols = _to_uniform_grid_at_fs(t_sec, y_cols_raw, fs_est)
                if t_uni.size == 0:
                    continue
                y_target_cols = _lpf_downsample_to_target(y_uni_cols, fs_est, self.target_sampling_rate)
                n_target = len(y_target_cols[0])
                t_target = np.arange(n_target, dtype=np.float64) / self.target_sampling_rate
            else:
                # 直接 目標レート等間隔へ線形補間（内側トリム）
                dt = 1.0 / self.target_sampling_rate
                t0 = np.ceil(t_sec[0] / dt) * dt
                t1 = np.floor(t_sec[-1] / dt) * dt
                if t1 <= t0:
                    continue
                n = int(round((t1 - t0) / dt)) + 1
                t_target = t0 + np.arange(n, dtype=np.float64) * dt
                y_target_cols = [np.interp(t_target, t_sec, y).astype(np.float32) for y in y_cols_raw]

            # 1秒窓（非オーバーラップ）
            win_len = int(round(self.target_sampling_rate * 1.0))
            total = t_target.size
            n_win = total // win_len
            if n_win == 0:
                continue
            n_use = n_win * win_len

            y_target_cols = [y[:n_use] for y in y_target_cols]
            t_target = t_target[:n_use]

            sig = np.stack(y_target_cols, axis=1).reshape(n_win, win_len, len(COLS_IMU)).astype(np.float32)
            t_w0 = t_target.reshape(n_win, win_len)[:, 0].astype(np.float32)

            windows.append(sig)
            t0_list.append(t_w0)

        if not windows:
            return np.empty((0, self.window_size, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)

        signals = np.concatenate(windows, axis=0)  # (N, W, 6)
        start_time_sec = np.concatenate(t0_list, axis=0)  # (N,)

        # 必要に応じてウィンドウサイズ/ストライドを再整形（既定は1秒窓/非オーバーラップのため多くの場合一致）
        if signals.shape[1] != self.window_size:
            # 窓長不一致時は再スライス（安全策: 先頭 window_size に切り詰め）
            trim = min(self.window_size, signals.shape[1])
            signals = signals[:, :trim, :]

        return signals, start_time_sec

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        # 本前処理では load_raw_data 内でクリーニング済み
        return data

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]):
        """
        センサー/モダリティごとに分割。

        入力: {person_id: (windows[N, W, 6], t0[N])}
        出力: {person_id: {"Head/ACC": {"X": (N, 3, W), "Y": (N,)}, "Head/GYRO": {...}}}
        備考: Y には活動ラベルが無いため未定義クラス -1 を格納する
        """
        result: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}

        for person_id, (windows, t0) in data.items():
            # (N, W, 6) -> (N, 6, W)
            X = np.transpose(windows, (0, 2, 1)).astype(np.float16)

            person_dict: Dict[str, Dict[str, np.ndarray]] = {}

            for modality, (ch_start, ch_end) in self.modalities.items():
                key = f"{self.sensor_name}/{modality}"
                X_mod = X[:, ch_start:ch_end, :]  # (N, 3, W)
                person_dict[key] = {
                    'X': X_mod,
                    'Y': np.full((X_mod.shape[0],), -1, dtype=np.int32),
                }

            result[person_id] = person_dict

        return result

    def save_processed_data(self, data) -> None:
        base_path = self.processed_data_path / self.dataset_name
        base_path.mkdir(parents=True, exist_ok=True)

        import json

        total_stats = {
            'dataset': self.dataset_name,
            'sensor_names': [self.sensor_name],
            'modalities': list(self.modalities.keys()),
            'target_sampling_rate': self.target_sampling_rate,
            'window_size': self.window_size,
            'stride': self.stride,
            'note': 'Y.npy stores -1 as undefined activity label (no labels provided).',
            'users': {},
        }

        for person_id, sensor_modality_data in data.items():
            user_name = f"USER{person_id:05d}"
            user_path = base_path / user_name
            user_path.mkdir(parents=True, exist_ok=True)

            user_stats = {'sensor_modalities': {}}

            for sensor_modality_name, arrays in sensor_modality_data.items():
                sensor_modality_path = user_path / sensor_modality_name
                sensor_modality_path.mkdir(parents=True, exist_ok=True)

                X = arrays['X']  # (N, 3, W)
                Y = arrays['Y']  # (N,) start_time_sec

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': list(X.shape),
                    'Y_shape': list(Y.shape),
                    'num_windows': int(len(Y)),
                }

                logger.info(f"Saved {user_name}/{sensor_modality_name}: X={X.shape}, Y={Y.shape}")

            total_stats['users'][user_name] = user_stats

        stats_path = base_path / 'dataset_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(total_stats, f, indent=2)
        logger.info(f"Saved dataset statistics to {stats_path}")


