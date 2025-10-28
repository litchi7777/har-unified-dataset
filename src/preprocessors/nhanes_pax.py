"""
NHANES PAX-Gデータの前処理

CDC FTPサーバーから1ユーザーずつダウンロードして処理し、
処理後はファイルを削除してストレージを節約します。
"""

import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import urllib.request
from urllib.error import URLError, HTTPError
import time
import logging

import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from .base import BasePreprocessor
from . import register_preprocessor

logger = logging.getLogger(__name__)


class NHANESPAXProcessor:
    """NHANES PAX-Gデータの前処理クラス"""
    
    def __init__(
        self,
        output_base: str = "/mnt/home/processed_data/NHANES_PAX",
        window_size: int = 5,  # seconds
        sampling_rate: int = 80,  # Hz
        target_sampling_rate: Optional[int] = None,  # Hz (Noneの場合はリサンプリングしない)
        std_threshold: float = 0.02,  # 標準偏差の合計の閾値
        temp_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        """
        Args:
            output_base: 処理済みデータの保存先ディレクトリ
            window_size: ウィンドウサイズ（秒）
            sampling_rate: 元のサンプリングレート（Hz）- NHANES PAXは80Hz
            target_sampling_rate: 目標サンプリングレート（Hz）- Noneの場合はリサンプリングしない
            std_threshold: アクティビティ判定用の標準偏差閾値
            temp_dir: 一時ファイル用ディレクトリ（Noneの場合はシステムデフォルト）
            max_retries: ダウンロード失敗時の最大リトライ回数
            retry_delay: リトライ間隔（秒）
        """
        self.output_base = Path(output_base)
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.target_sampling_rate = target_sampling_rate if target_sampling_rate is not None else sampling_rate
        self.std_threshold = std_threshold
        self.temp_dir = temp_dir
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # ウィンドウ長（サンプル数）- リサンプリング後のレートで計算
        self.window_length = window_size * self.target_sampling_rate
        
        # 出力ディレクトリを作成
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        # FTPサーバーのベースURL（PAX-GとPAX-Hの2つのソース）
        self.ftp_base_g = "https://ftp.cdc.gov/pub/pax_g/"
        self.ftp_base_h = "https://ftp.cdc.gov/pub/pax_h/"
        self.pax_h_start_id = 73557  # PAX-Hの開始ID
        
        # 進捗管理ファイル
        self.progress_file = self.output_base / "processing_progress.json"
        self.failed_users_file = self.output_base / "failed_users.json"
    
    def check_user_exists(self, user_id: int) -> bool:
        """
        FTPサーバーで特定ユーザーのファイルが存在するかチェック
        
        Args:
            user_id: ユーザーID
            
        Returns:
            ファイルが存在するかどうか
        """
        import urllib.request
        from urllib.error import URLError, HTTPError
        
        # PAX-GとPAX-Hのどちらかに存在するかチェック
        if user_id < self.pax_h_start_id:
            url = f"{self.ftp_base_g}{user_id}.tar.bz2"
        else:
            url = f"{self.ftp_base_h}{user_id}.tar.bz2"
            
        try:
            # HEADリクエストでファイルの存在確認（ダウンロードしない）
            req = urllib.request.Request(url)
            req.get_method = lambda: 'HEAD'
            response = urllib.request.urlopen(req, timeout=10)
            return response.status == 200
        except HTTPError as e:
            return e.code == 200
        except (URLError, Exception):
            return False
    
    def discover_available_users(self, start_id: int = 62161, end_id: int = 72161, 
                               batch_size: int = 100) -> List[int]:
        """
        利用可能なユーザーIDを自動発見
        
        Args:
            start_id: 開始ID
            end_id: 終了ID
            batch_size: 一度にチェックするID数
            
        Returns:
            実際に存在するユーザーIDのリスト
        """
        print(f"Discovering available users from {start_id} to {end_id}...")
        
        available_users = []
        total_range = end_id - start_id + 1
        
        # バッチごとに確認
        for batch_start in tqdm(range(start_id, end_id + 1, batch_size), 
                               desc="Checking user availability"):
            batch_end = min(batch_start + batch_size - 1, end_id)
            batch_ids = list(range(batch_start, batch_end + 1))
            
            # 並列でチェック
            from multiprocessing import Pool
            with Pool(processes=4) as pool:
                results = pool.map(self.check_user_exists, batch_ids)
            
            # 存在するIDを追加
            for user_id, exists in zip(batch_ids, results):
                if exists:
                    available_users.append(user_id)
        
        print(f"Found {len(available_users)} available users out of {total_range} checked")
        if available_users:
            print(f"Available user IDs: {sorted(available_users)}")
        return available_users
    
    def get_user_ids(self, start_id: int = 62161, end_id: int = 72161, 
                    discover: bool = False) -> List[int]:
        """
        処理対象のユーザーIDリストを生成
        
        Args:
            start_id: 開始ID
            end_id: 終了ID
            discover: 実際に存在するユーザーIDのみを取得するか
            
        Returns:
            ユーザーIDのリスト
        """
        if discover:
            return self.discover_available_users(start_id, end_id)
        else:
            return list(range(start_id, end_id + 1))
    
    def save_progress(self, processed_users: List[int], failed_users: List[int], 
                     current_batch: int = 0, total_batches: int = 0) -> None:
        """
        進捗状況を保存
        
        Args:
            processed_users: 処理済みユーザーIDのリスト
            failed_users: 失敗したユーザーIDのリスト
            current_batch: 現在のバッチ番号
            total_batches: 総バッチ数
        """
        import json
        from datetime import datetime
        
        progress_data = {
            "last_updated": datetime.now().isoformat(),
            "processed_users": processed_users,
            "failed_users": failed_users,
            "current_batch": current_batch,
            "total_batches": total_batches,
            "total_processed": len(processed_users),
            "total_failed": len(failed_users)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self) -> Tuple[List[int], List[int], int, int]:
        """
        進捗状況を読み込み
        
        Returns:
            (processed_users, failed_users, current_batch, total_batches)
        """
        import json
        
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                return (
                    data.get("processed_users", []),
                    data.get("failed_users", []),
                    data.get("current_batch", 0),
                    data.get("total_batches", 0)
                )
            except Exception as e:
                print(f"Warning: Could not load progress file: {e}")
        
        return [], [], 0, 0
    
    def get_remaining_users(self, all_users: List[int]) -> List[int]:
        """
        処理が必要な残りのユーザーIDを取得
        
        Args:
            all_users: 全ユーザーIDのリスト
            
        Returns:
            処理が必要なユーザーIDのリスト
        """
        processed_users, failed_users, _, _ = self.load_progress()
        completed_users = set(processed_users + failed_users)
        
        # ファイルの存在も確認
        remaining_users = []
        for user_id in all_users:
            if user_id not in completed_users:
                output_path = self.output_base / "nhanes" / f"USER{user_id}" / "PAX" / "ACC" / "X.npy"
                if not output_path.exists():
                    remaining_users.append(user_id)
                else:
                    # ファイルが存在するが進捗に記録されていない場合は記録
                    processed_users.append(user_id)
        
        # 進捗を更新
        if len(processed_users) != len(set(processed_users)):
            processed_users = list(set(processed_users))
            self.save_progress(processed_users, failed_users)
        
        return remaining_users
    
    def download_user_data(self, user_id: int, temp_path: Path) -> bool:
        """
        特定ユーザーのデータをダウンロード
        
        Args:
            user_id: ユーザーID
            temp_path: ダウンロード先の一時パス
            
        Returns:
            ダウンロード成功の可否
        """
        # PAX-GとPAX-Hのどちらかからダウンロード
        if user_id < self.pax_h_start_id:
            url = f"{self.ftp_base_g}{user_id}.tar.bz2"
        else:
            url = f"{self.ftp_base_h}{user_id}.tar.bz2"
            
        download_path = temp_path / f"{user_id}.tar.bz2"
        
        for attempt in range(self.max_retries):
            try:
                # ダウンロード進行状況用のpbar
                pbar = None
                
                def show_progress(block_num, block_size, total_size):
                    nonlocal pbar
                    if pbar is None:
                        pbar = tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc=f"Downloading user {user_id}"
                        )
                    downloaded = block_num * block_size
                    if downloaded < total_size:
                        pbar.update(block_size)
                    else:
                        pbar.update(total_size - pbar.n)
                        pbar.close()
                
                # ダウンロード実行（進行状況表示付き）
                urllib.request.urlretrieve(url, download_path, reporthook=show_progress)
                
                if pbar and not pbar.disable:
                    pbar.close()
                    
                return True
                
            except HTTPError as e:
                if pbar and not pbar.disable:
                    pbar.close()

                # 404エラーの場合は即座に失敗（ファイルが存在しない）
                if e.code == 404:
                    return False

                # その他のHTTPエラーはリトライ
                if attempt < self.max_retries - 1:
                    print(f"Download failed for user {user_id}, attempt {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to download user {user_id} after {self.max_retries} attempts: {e}")
                    return False
            except URLError as e:
                if pbar and not pbar.disable:
                    pbar.close()

                # ネットワークエラーはリトライ
                if attempt < self.max_retries - 1:
                    print(f"Download failed for user {user_id}, attempt {attempt + 1}/{self.max_retries}: {e}")
                    time.sleep(self.retry_delay)
                else:
                    print(f"Failed to download user {user_id} after {self.max_retries} attempts: {e}")
                    return False
            except Exception as e:
                if pbar and not pbar.disable:
                    pbar.close()
                    
                print(f"Unexpected error downloading user {user_id}: {e}")
                return False
        
        return False
    
    def extract_archive(self, archive_path: Path, extract_path: Path) -> bool:
        """
        tar.bz2アーカイブを解凍
        
        Args:
            archive_path: アーカイブファイルのパス
            extract_path: 解凍先のパス
            
        Returns:
            解凍成功の可否
        """
        try:
            with tarfile.open(archive_path, 'r:bz2') as tar:
                tar.extractall(extract_path)
            return True
        except Exception as e:
            print(f"Failed to extract {archive_path}: {e}")
            return False
    
    def process_sensor_file(self, file_path: Path) -> Optional[np.ndarray]:
        """
        個別のセンサーCSVファイルを処理（リサンプリング含む）

        Args:
            file_path: CSVファイルのパス

        Returns:
            処理されたセンサーデータ（3チャンネル×サンプル数）
        """
        try:
            # CSVを読み込み
            df = pd.read_csv(
                file_path,
                names=["timestamp", "X", "Y", "Z"],
                skiprows=1,
                on_bad_lines='skip',
                dtype={"timestamp": str, "X": float, "Y": float, "Z": float}
            )

            # 数値に変換（エラーはNaNに）
            x = pd.to_numeric(df["X"], errors='coerce').values
            y = pd.to_numeric(df["Y"], errors='coerce').values
            z = pd.to_numeric(df["Z"], errors='coerce').values

            # NaNを除外
            valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask]

            if len(x) == 0:
                return None

            # リサンプリング（必要な場合）
            if self.target_sampling_rate != self.sampling_rate:
                # ポリフェーズフィルタリングでリサンプリング（他のデータセットと同じ方法）
                from math import gcd

                # 比率を簡約化（例: 80Hz -> 30Hz = up=3, down=8）
                multiplier = 1000
                up = int(self.target_sampling_rate * multiplier)
                down = int(self.sampling_rate * multiplier)
                common_divisor = gcd(up, down)
                up = up // common_divisor
                down = down // common_divisor

                # 各チャンネルをポリフェーズフィルタリングでリサンプリング
                x = signal.resample_poly(x, up, down)
                y = signal.resample_poly(y, up, down)
                z = signal.resample_poly(z, up, down)

            # (3, samples) の形状で返す、float16で効率化
            result = np.stack([x, y, z], axis=0).astype(np.float16)
            return result

        except Exception as e:
            print(f"Error processing sensor file {file_path}: {e}")
            return None
    
    def extract_valid_windows(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        データから閾値以上のアクティビティがあるウィンドウを抽出
        
        Args:
            data: センサーデータ（3チャンネル×サンプル数）
            
        Returns:
            有効なウィンドウ（ウィンドウ数×3チャンネル×ウィンドウ長）
        """
        total_length = data.shape[1]
        n_windows = total_length // self.window_length
        
        if n_windows == 0:
            return None
        
        # ウィンドウに分割
        segments = data[:, :n_windows * self.window_length].reshape(
            3, n_windows, self.window_length
        )
        
        # 各ウィンドウの標準偏差を計算
        stds = np.std(segments, axis=2)  # shape: (3, n_windows)
        std_sum = np.sum(stds, axis=0)   # shape: (n_windows,)
        
        # 閾値以上のウィンドウを抽出
        valid_mask = std_sum >= self.std_threshold
        valid_segments = segments[:, valid_mask, :]  # shape: (3, n_valid, window_length)
        
        if valid_segments.shape[1] > 0:
            # (n_valid, 3, window_length) の形状に変換
            return valid_segments.transpose(1, 0, 2)
        else:
            return None
    
    def process_user(self, user_id: int) -> Tuple[bool, str]:
        """
        1ユーザーのデータを処理（ダウンロード→処理→保存→削除）
        
        Args:
            user_id: ユーザーID
            
        Returns:
            (成功フラグ, メッセージ)
        """
        # 既に処理済みかチェック（X.npyで確認）
        output_dir = self.output_base / "nhanes" / f"USER{user_id}" / "PAX" / "ACC"
        data_path = output_dir / "X.npy"
        if data_path.exists():
            return True, f"User {user_id} already processed, skipping"
        
        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory(dir=self.temp_dir) as temp_dir:
            temp_path = Path(temp_dir)
            
            # 1. ダウンロード
            if not self.download_user_data(user_id, temp_path):
                return False, f"Failed to download user {user_id}"
            
            # 2. 解凍
            archive_path = temp_path / f"{user_id}.tar.bz2"
            extract_path = temp_path / "extracted"
            extract_path.mkdir(exist_ok=True)
            
            if not self.extract_archive(archive_path, extract_path):
                return False, f"Failed to extract user {user_id}"
            
            # アーカイブは不要になったので削除
            archive_path.unlink()
            
            # 3. センサーファイルを処理
            sensor_files = list(extract_path.glob("**/*.sensor.csv"))
            if not sensor_files:
                return False, f"No sensor files found for user {user_id}"
            
            all_segments = []
            for sensor_file in sensor_files:
                # センサーデータを読み込み
                data = self.process_sensor_file(sensor_file)
                if data is None:
                    continue
                
                # 有効なウィンドウを抽出
                valid_windows = self.extract_valid_windows(data)
                if valid_windows is not None:
                    all_segments.append(valid_windows)
            
            # 4. データを保存
            if all_segments:
                # すべてのセグメントを結合
                user_data = np.concatenate(all_segments, axis=0)

                # float16で保存（メモリ効率化）
                user_data = user_data.astype(np.float16)

                # ラベルを生成（全て-1、float16）
                labels = np.full(user_data.shape[0], -1, dtype=np.float16)

                # 保存ディレクトリを作成
                output_dir.mkdir(parents=True, exist_ok=True)

                # データを保存
                np.save(data_path, user_data)
                np.save(output_dir / "Y.npy", labels)

                return True, f"Successfully processed user {user_id}: {user_data.shape[0]} windows saved"
            else:
                return False, f"No valid windows found for user {user_id}"
    
    def process_all_users(
        self,
        user_ids: Optional[List[int]] = None,
        start_id: int = 62161,
        end_id: int = 72161,
        parallel: bool = False,
        n_workers: int = 4
    ):
        """
        すべてのユーザーを処理（進捗管理付き）
        
        Args:
            user_ids: 処理するユーザーIDのリスト（Noneの場合は範囲指定）
            start_id: 開始ID（user_idsがNoneの場合）
            end_id: 終了ID（user_idsがNoneの場合）
            parallel: 並列処理を使用するか
            n_workers: 並列処理時のワーカー数
        """
        # ユーザーIDリストを準備
        if user_ids is None:
            user_ids = self.get_user_ids(start_id, end_id)
        
        # 進捗状況を読み込み
        processed_users, failed_users, last_batch, total_batches = self.load_progress()
        
        # 残りのユーザーを取得
        remaining_users = self.get_remaining_users(user_ids)
        
        print(f"Total users: {len(user_ids)}")
        print(f"Already processed: {len(processed_users)}")
        print(f"Previously failed: {len(failed_users)}")
        print(f"Remaining to process: {len(remaining_users)}")
        
        if not remaining_users:
            print("All users have been processed!")
            return
        
        print(f"Resuming processing from user batch...")
        
        if parallel:
            # 並列処理（マルチプロセス）
            from multiprocessing import Pool
            
            with Pool(processes=n_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(self.process_user, remaining_users),
                    total=len(remaining_users),
                    desc="Processing remaining users"
                ))
        else:
            # 逐次処理（進捗保存付き）
            results = []
            for i, user_id in enumerate(tqdm(remaining_users, desc="Processing remaining users")):
                result = self.process_user(user_id)
                results.append(result)
                
                success, message = result
                if success and "already processed" not in message:
                    processed_users.append(user_id)
                    print(f"✓ {message}")
                else:
                    failed_users.append(user_id)
                    print(f"⚠️  {message}")
                
                # 定期的に進捗を保存（10ユーザーごと）
                if (i + 1) % 10 == 0:
                    self.save_progress(processed_users, failed_users)
                    print(f"Progress saved: {len(processed_users)} processed, {len(failed_users)} failed")
        
        # 最終的な進捗を保存
        if not parallel:
            self.save_progress(processed_users, failed_users)
        
        # 統計を表示
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        print(f"\n" + "="*60)
        print(f"Processing complete!")
        print(f"✓ Successful this session: {successful}/{len(results)}")
        print(f"✗ Failed this session: {failed}/{len(results)}")
        print(f"📊 Total processed: {len(processed_users)}/{len(user_ids)}")
        print(f"📊 Total failed: {len(failed_users)}/{len(user_ids)}")
        print("="*60)


@register_preprocessor('nhanes')
class NHANESPreprocessor(BasePreprocessor):
    """NHANES PAX-Gデータの前処理クラス（統合版）"""
    
    def get_dataset_name(self) -> str:
        return "nhanes"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # NHANES固有の設定（PAX-GとPAX-Hの2つのソース）
        self.ftp_base_g = "https://ftp.cdc.gov/pub/pax_g/"
        self.ftp_base_h = "https://ftp.cdc.gov/pub/pax_h/"
        self.pax_h_start_id = 73557  # PAX-Hの開始ID
        self.window_size = config.get('window_size', 5)  # seconds
        self.sampling_rate = config.get('sampling_rate', 80)  # Hz (元のサンプリングレート)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (リサンプリング先、デフォルト30Hz)
        self.std_threshold = config.get('std_threshold', 0.02)
        self.start_id = config.get('start_id', 62161)
        self.end_id = config.get('end_id', 62170)
        self.batch_size = config.get('batch_size', 100)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.parallel = config.get('parallel', True)
        self.n_workers = config.get('workers', 4)

        # 最終的なサンプリングレート
        final_rate = self.target_sampling_rate if self.target_sampling_rate is not None else self.sampling_rate

        # ウィンドウ長（サンプル数）- リサンプリング後のレートで計算
        self.window_length = self.window_size * final_rate

        # プロセッサーを初期化
        self.processor = NHANESPAXProcessor(
            output_base=str(self.processed_data_path),
            window_size=self.window_size,
            sampling_rate=self.sampling_rate,
            target_sampling_rate=self.target_sampling_rate,
            std_threshold=self.std_threshold,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay
        )
    
    def download_dataset(self) -> None:
        """
        データセットをダウンロード
        
        NHANESの場合は個別ユーザーごとにダウンロードして処理するため、
        ここでは何もしない（process_all_usersで実行）
        """
        logger.info("NHANES PAX-G dataset will be downloaded automatically during processing")
    
    def load_raw_data(self) -> List[int]:
        """
        処理対象のユーザーIDリストを返す
        
        Returns:
            ユーザーIDのリスト
        """
        # 設定で発見モードが有効かチェック
        discover_mode = self.config.get('discover_users', False)
        
        if discover_mode:
            logger.info("User discovery mode enabled. Scanning for available users...")
            user_ids = self.processor.discover_available_users(
                start_id=self.start_id,
                end_id=self.end_id,
                batch_size=self.config.get('discovery_batch_size', 100)
            )
            logger.info(f"Discovered {len(user_ids)} available users: {sorted(user_ids)}")
        else:
            user_ids = list(range(self.start_id, self.end_id + 1))
            logger.info(f"Using ID range mode: {len(user_ids)} users ({self.start_id} to {self.end_id})")
        
        return user_ids
    
    def clean_data(self, user_ids: List[int]) -> List[int]:
        """
        データのクリーニング（既に処理済みのユーザーを除外）
        
        Args:
            user_ids: 全ユーザーIDのリスト
            
        Returns:
            処理対象のユーザーIDのリスト
        """
        # 進捗管理機能を使用
        remaining_users = self.processor.get_remaining_users(user_ids)
        
        already_processed = len(user_ids) - len(remaining_users)
        if already_processed > 0:
            logger.info(f"Skipping {already_processed} already processed users")
        
        logger.info(f"Remaining users to process: {len(remaining_users)}")
        return remaining_users
    
    def extract_features(self, user_ids: List[int]) -> Dict[str, Any]:
        """
        特徴抽出（ユーザーデータの処理）
        
        Args:
            user_ids: 処理対象のユーザーIDのリスト
            
        Returns:
            処理結果の統計情報
        """
        if not user_ids:
            logger.info("No users to process")
            return {"processed_users": 0, "total_users": 0}
        
        # バッチごとに処理
        total_users = len(user_ids)
        processed_users = 0
        failed_users = 0
        
        for i in range(0, len(user_ids), self.batch_size):
            batch_users = user_ids[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(user_ids) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_users)} users)")
            
            # バッチ処理
            if self.parallel:
                from multiprocessing import Pool
                
                with Pool(processes=self.n_workers) as pool:
                    results = list(tqdm(
                        pool.imap_unordered(self.processor.process_user, batch_users),
                        total=len(batch_users),
                        desc=f"Batch {batch_num}"
                    ))
            else:
                results = []
                for user_id in tqdm(batch_users, desc=f"Batch {batch_num}"):
                    results.append(self.processor.process_user(user_id))
            
            # 結果を集計
            batch_success = sum(1 for success, _ in results if success)
            batch_failed = len(results) - batch_success
            
            processed_users += batch_success
            failed_users += batch_failed
            
            logger.info(f"Batch {batch_num} complete: {batch_success}/{len(batch_users)} succeeded")
        
        # 統計情報を返す
        return {
            "total_users": total_users,
            "processed_users": processed_users,
            "failed_users": failed_users,
            "success_rate": processed_users / total_users if total_users > 0 else 0
        }
    
    def save_processed_data(self, stats: Dict[str, Any]) -> None:
        """
        処理済みデータの保存（統計情報の保存）
        
        Args:
            stats: 処理結果の統計情報
        """
        # 統計情報をJSONファイルに保存
        import json
        
        stats_file = self.processed_data_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Processing statistics saved to {stats_file}")
    
    def preprocess(self) -> None:
        """
        前処理のメイン処理
        """
        logger.info("Starting NHANES PAX-G preprocessing...")
        
        # 1. 処理対象のユーザーIDリストを取得
        user_ids = self.load_raw_data()
        
        # 2. データクリーニング（既に処理済みを除外）
        remaining_users = self.clean_data(user_ids)
        
        # 3. 特徴抽出（ダウンロード→処理→保存）
        stats = self.extract_features(remaining_users)
        
        # 4. 統計情報を保存
        self.save_processed_data(stats)
        
        logger.info("NHANES PAX-G preprocessing completed!")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        処理統計を取得
        
        Returns:
            統計情報の辞書
        """
        stats_file = self.processed_data_path / "processing_stats.json"
        
        if stats_file.exists():
            import json
            with open(stats_file, 'r') as f:
                return json.load(f)
        else:
            # 統計ファイルがない場合は処理済みファイル数をカウント
            processed_count = len(list(self.processed_data_path.glob("nhanes/USER*/PAX/ACC/X.npy")))
            return {
                "processed_users": processed_count,
                "total_users": self.end_id - self.start_id + 1
            }