#!/usr/bin/env python3
"""
既存の処理済みデータからmetadata.jsonを再構築

使用例:
    python rebuild_metadata.py --dataset paal
    python rebuild_metadata.py --dataset selfback
    python rebuild_metadata.py --all
"""

import argparse
import json
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scan_dataset_structure(dataset_path: Path) -> dict:
    """
    データセットディレクトリをスキャンしてusers辞書を構築

    Args:
        dataset_path: データセットのパス (e.g., data/processed/paal)

    Returns:
        users辞書 {user_id: {sensor_modalities: {...}}}
    """
    users = {}

    # USER*ディレクトリを探す
    user_dirs = sorted(dataset_path.glob("USER*"))

    for user_dir in user_dirs:
        if not user_dir.is_dir():
            continue

        user_id = user_dir.name
        user_data = {"sensor_modalities": {}}

        # センサー/モダリティの組み合わせを探す
        for sensor_dir in user_dir.iterdir():
            if not sensor_dir.is_dir():
                continue

            sensor_name = sensor_dir.name

            for modality_dir in sensor_dir.iterdir():
                if not modality_dir.is_dir():
                    continue

                modality_name = modality_dir.name
                sensor_modality_key = f"{sensor_name}/{modality_name}"

                # X.npy, Y.npy, labels.npyを確認
                x_path = modality_dir / "X.npy"
                y_path = modality_dir / "Y.npy"
                labels_path = modality_dir / "labels.npy"

                if x_path.exists() and labels_path.exists():
                    # ファイルサイズと形状を取得
                    X = np.load(x_path)
                    Y = None
                    if y_path.exists():
                        Y = np.load(y_path)
                    labels = np.load(labels_path)

                    # X, Y, Zの形状を記録
                    modality_info = {
                        "X_shape": list(X.shape),
                        "Y_shape": list(Y.shape) if Y is not None else None,
                        "labels_shape": list(labels.shape),
                        "num_windows": len(labels),
                        "unique_labels": sorted([int(l) for l in np.unique(labels)])
                    }

                    # Z.npyがある場合
                    z_path = modality_dir / "Z.npy"
                    if z_path.exists():
                        Z = np.load(z_path)
                        modality_info["Z_shape"] = list(Z.shape)

                    user_data["sensor_modalities"][sensor_modality_key] = modality_info

        if user_data["sensor_modalities"]:
            users[user_id] = user_data

    return users


def rebuild_metadata(dataset_path: Path, overwrite: bool = False):
    """
    データセットのmetadata.jsonを再構築

    Args:
        dataset_path: データセットのパス
        overwrite: 既存のmetadataを上書きするか
    """
    metadata_path = dataset_path / "metadata.json"

    # 既存のmetadataを読み込む
    existing_metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            existing_metadata = json.load(f)

        # すでにusersキーがある場合
        if "users" in existing_metadata and not overwrite:
            logger.info(f"{dataset_path.name}: metadata already has 'users' key. Skipping.")
            return

    logger.info(f"{dataset_path.name}: Scanning directory structure...")

    # ディレクトリ構造をスキャン
    users = scan_dataset_structure(dataset_path)

    if not users:
        logger.warning(f"{dataset_path.name}: No users found. Skipping.")
        return

    # 統計情報を計算
    total_windows = sum(
        sum(
            mod_info['num_windows']
            for mod_info in user_data['sensor_modalities'].values()
        )
        for user_data in users.values()
    )

    # すべてのセンサー位置を収集
    all_sensors = set()
    for user_data in users.values():
        for sensor_modality in user_data['sensor_modalities'].keys():
            sensor_name = sensor_modality.split('/')[0]
            all_sensors.add(sensor_name)

    # 新しいmetadataを構築
    new_metadata = {
        "dataset": dataset_path.name,
        **existing_metadata,  # 既存のフィールドを保持
        "users": users,
        "num_users": len(users),
    }

    # num_windowsを更新（計算した値を使用）
    if "num_windows" in new_metadata:
        if new_metadata["num_windows"] != total_windows:
            logger.warning(
                f"{dataset_path.name}: num_windows mismatch: "
                f"metadata={new_metadata['num_windows']}, calculated={total_windows}"
            )
            new_metadata["num_windows"] = total_windows

    # sensor_namesを追加（存在しない場合）
    if "sensor_names" not in new_metadata:
        new_metadata["sensor_names"] = sorted(all_sensors)

    # 保存
    with open(metadata_path, 'w') as f:
        json.dump(new_metadata, f, indent=2)

    logger.info(
        f"{dataset_path.name}: Rebuilt metadata with {len(users)} users, "
        f"{total_windows} windows"
    )
    logger.info(f"  Sensors: {sorted(all_sensors)}")
    logger.info(f"  Saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Rebuild metadata.json for datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name to rebuild (e.g., paal, selfback)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild all datasets"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing users data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data directory"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    if args.all:
        # すべてのデータセットを処理
        for dataset_path in sorted(data_dir.iterdir()):
            if dataset_path.is_dir():
                try:
                    rebuild_metadata(dataset_path, overwrite=args.overwrite)
                except Exception as e:
                    logger.error(f"Error processing {dataset_path.name}: {e}")

    elif args.dataset:
        # 特定のデータセットのみ処理
        dataset_path = data_dir / args.dataset
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return

        rebuild_metadata(dataset_path, overwrite=args.overwrite)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
