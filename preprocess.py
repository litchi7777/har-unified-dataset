"""
データセット前処理のエントリーポイント

複数のデータセットに対する前処理を実行

使用例:
    # 単一データセットを処理
    python preprocess.py --dataset dsads

    # ダウンロードから前処理まで一括実行
    python preprocess.py --dataset dsads --download

    # 複数のデータセットを処理
    python preprocess.py --dataset dsads opportunity pamap2

    # カスタム設定ファイルを使用
    python preprocess.py --dataset dsads --config configs/my_preprocess.yaml

    # 利用可能なデータセットを表示
    python preprocess.py --list
"""

import argparse
import yaml
import logging
from pathlib import Path
import sys

from src.preprocessors import get_preprocessor, list_preprocessors


def setup_logging(config: dict) -> None:
    """
    ロギングの設定

    Args:
        config: ロギング設定を含む辞書
    """
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 基本設定
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # ファイルハンドラを追加（オプション）
    if log_config.get('save_to_file', False):
        log_dir = Path(log_config.get('log_dir', 'logs/preprocessing'))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'preprocess.log'

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str))
        logging.getLogger().addHandler(file_handler)

        logging.info(f"Logging to file: {log_file}")


def load_config(config_path: str) -> dict:
    """
    設定ファイルを読み込む

    Args:
        config_path: 設定ファイルのパス

    Returns:
        設定辞書
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def preprocess_dataset(dataset_name: str, config: dict, download: bool = False) -> None:
    """
    単一のデータセットを前処理

    Args:
        dataset_name: データセット名
        config: 設定辞書
        download: データセットをダウンロードするか
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info(f"Starting preprocessing for dataset: {dataset_name}")
    logger.info("=" * 80)

    # データセット固有の設定を取得
    dataset_config = config.get('datasets', {}).get(dataset_name, {})

    # グローバル設定をマージ
    global_config = config.get('global', {})
    dataset_config = {**global_config, **dataset_config}

    # プリプロセッサを取得
    try:
        PreprocessorClass = get_preprocessor(dataset_name)
    except KeyError as e:
        logger.error(str(e))
        raise

    # 前処理を実行
    try:
        preprocessor = PreprocessorClass(dataset_config)

        # ダウンロード（オプション）
        if download:
            logger.info("Downloading dataset...")
            try:
                preprocessor.download_dataset()
            except NotImplementedError as e:
                logger.error(str(e))
                raise

        # 前処理
        preprocessor.preprocess()

        # 統計情報を表示
        stats = preprocessor.get_statistics()
        logger.info("Preprocessing statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        logger.info(f"Successfully completed preprocessing for {dataset_name}")

    except Exception as e:
        logger.error(f"Error during preprocessing of {dataset_name}: {e}", exc_info=True)
        raise


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description='Preprocess HAR datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess DSADS dataset (data must already exist)
  python preprocess.py --dataset dsads

  # Download and preprocess DSADS dataset
  python preprocess.py --dataset dsads --download

  # Preprocess multiple datasets
  python preprocess.py --dataset dsads opportunity pamap2

  # Use custom config file
  python preprocess.py --dataset dsads --config configs/my_preprocess.yaml

  # List available datasets
  python preprocess.py --list
        """
    )

    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        help='Dataset name(s) to preprocess (e.g., dsads, opportunity, pamap2)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/preprocess.yaml',
        help='Path to config file (default: configs/preprocess.yaml)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset before preprocessing'
    )

    args = parser.parse_args()

    # 利用可能なデータセットを表示
    if args.list:
        available = list_preprocessors()
        print("Available datasets:")
        for dataset in available:
            print(f"  - {dataset}")
        return

    # データセット名が指定されていない場合
    if not args.dataset:
        parser.print_help()
        print("\nError: Please specify at least one dataset with --dataset")
        sys.exit(1)

    # 設定ファイルを読み込む
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ロギングを設定
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info(f"Config file: {args.config}")
    logger.info(f"Datasets to preprocess: {args.dataset}")
    if args.download:
        logger.info("Download mode: enabled")

    # 各データセットを前処理
    success_count = 0
    failed_datasets = []

    for dataset_name in args.dataset:
        # データセット名を小文字に変換（大文字小文字を気にしなくて良いように）
        dataset_name_lower = dataset_name.lower()
        try:
            preprocess_dataset(dataset_name_lower, config, download=args.download)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to preprocess {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # 結果サマリー
    logger.info("=" * 80)
    logger.info("Preprocessing Summary")
    logger.info("=" * 80)
    logger.info(f"Total datasets: {len(args.dataset)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_datasets)}")

    if failed_datasets:
        logger.error(f"Failed datasets: {', '.join(failed_datasets)}")
        sys.exit(1)
    else:
        logger.info("All datasets preprocessed successfully!")


if __name__ == '__main__':
    main()
