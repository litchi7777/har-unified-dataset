"""
基底前処理クラス
すべてのデータセット固有の前処理クラスはこのクラスを継承する
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BasePreprocessor(ABC):
    """
    データセット前処理の基底クラス

    すべてのデータセット固有の前処理クラスはこのクラスを継承し、
    必要なメソッドを実装する必要があります。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 前処理設定（YAMLから読み込まれた辞書）
        """
        self.config = config
        self.dataset_name = self.get_dataset_name()
        self.raw_data_path = Path(config.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(config.get('processed_data_path', 'data/processed'))
        self.setup_paths()

    @abstractmethod
    def get_dataset_name(self) -> str:
        """
        データセット名を返す

        Returns:
            データセット名（例: 'dsads', 'opportunity'）
        """
        pass

    def download_dataset(self) -> None:
        """
        データセットをダウンロード（オプション）

        各データセットで必要に応じて実装。
        実装しない場合は NotImplementedError を発生させる。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement download_dataset(). "
            "Please download the dataset manually or implement this method."
        )

    @abstractmethod
    def load_raw_data(self) -> Any:
        """
        生データを読み込む

        Returns:
            読み込んだ生データ（形式はデータセットによって異なる）
        """
        pass

    @abstractmethod
    def clean_data(self, data: Any) -> Any:
        """
        データのクリーニング処理

        Args:
            data: 生データ

        Returns:
            クリーニング済みデータ
        """
        pass

    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """
        特徴抽出処理

        Args:
            data: クリーニング済みデータ

        Returns:
            特徴抽出済みデータ
        """
        pass

    @abstractmethod
    def save_processed_data(self, data: Any) -> None:
        """
        処理済みデータを保存

        Args:
            data: 処理済みデータ
        """
        pass

    def setup_paths(self) -> None:
        """
        必要なディレクトリを作成
        """
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

        # データセット固有のディレクトリ
        dataset_processed_path = self.processed_data_path / self.dataset_name
        dataset_processed_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Setup paths for {self.dataset_name}")
        logger.info(f"  Raw data: {self.raw_data_path}")
        logger.info(f"  Processed data: {dataset_processed_path}")

    def validate_raw_data(self) -> bool:
        """
        生データが存在するか検証

        Returns:
            生データが存在する場合True
        """
        if not self.raw_data_path.exists():
            logger.error(f"Raw data path does not exist: {self.raw_data_path}")
            return False
        return True

    def preprocess(self) -> None:
        """
        前処理パイプライン全体を実行

        このメソッドは通常オーバーライド不要。
        各ステップのメソッドをオーバーライドすることで前処理をカスタマイズ。
        """
        logger.info(f"Starting preprocessing for {self.dataset_name}")

        # 1. 生データの検証
        if not self.validate_raw_data():
            raise FileNotFoundError(f"Raw data not found for {self.dataset_name}")

        # 2. データ読み込み
        logger.info("Loading raw data...")
        raw_data = self.load_raw_data()

        # 3. データクリーニング
        logger.info("Cleaning data...")
        cleaned_data = self.clean_data(raw_data)

        # 4. 特徴抽出
        logger.info("Extracting features...")
        processed_data = self.extract_features(cleaned_data)

        # 5. 保存
        logger.info("Saving processed data...")
        self.save_processed_data(processed_data)

        logger.info(f"Preprocessing completed for {self.dataset_name}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        処理済みデータの統計情報を取得（オプション）

        Returns:
            統計情報の辞書
        """
        return {
            "dataset": self.dataset_name,
            "raw_data_path": str(self.raw_data_path),
            "processed_data_path": str(self.processed_data_path),
        }
