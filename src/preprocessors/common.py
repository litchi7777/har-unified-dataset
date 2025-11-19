"""
データセットダウンロード・解凍の共通ユーティリティ
"""

import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, desc: Optional[str] = None) -> None:
    """
    ファイルをダウンロード（プログレスバー付き）

    Args:
        url: ダウンロードURL
        output_path: 保存先パス
        desc: プログレスバーの説明文
    """
    if desc is None:
        desc = 'Downloading'

    logger.info(f"Downloading from {url}")
    logger.info(f"Saving to {output_path}")

    # ディレクトリ作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ダウンロード
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # ファイルサイズを取得
    total_size = int(response.headers.get('content-length', 0))

    # プログレスバー付きでダウンロード
    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Download complete: {output_path}")


def extract_zip(zip_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    ZIPファイルを解凍

    Args:
        zip_path: ZIPファイルのパス
        extract_to: 解凍先ディレクトリ
        desc: プログレスバーの説明文

    Returns:
        解凍先のパス
    """
    if desc is None:
        desc = 'Extracting'

    logger.info(f"Extracting {zip_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # ファイル数を取得
        file_count = len(zip_ref.namelist())

        # プログレスバー付きで解凍
        with tqdm(total=file_count, desc=desc) as pbar:
            for file in zip_ref.namelist():
                zip_ref.extract(file, extract_to)
                pbar.update(1)

    logger.info(f"Extraction complete: {extract_to}")
    return extract_to


def extract_tar(tar_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    TAR/TGZファイルを解凍

    Args:
        tar_path: TARファイルのパス
        extract_to: 解凍先ディレクトリ
        desc: プログレスバーの説明文

    Returns:
        解凍先のパス
    """
    if desc is None:
        desc = 'Extracting'

    logger.info(f"Extracting {tar_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, 'r:*') as tar_ref:
        # ファイル数を取得
        members = tar_ref.getmembers()
        file_count = len(members)

        # プログレスバー付きで解凍
        with tqdm(total=file_count, desc=desc) as pbar:
            for member in members:
                tar_ref.extract(member, extract_to)
                pbar.update(1)

    logger.info(f"Extraction complete: {extract_to}")
    return extract_to


def extract_rar(rar_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    RARアーカイブを解凍（unrarコマンドを使用）

    Args:
        rar_path: RARファイルのパス
        extract_to: 解凍先ディレクトリ
        desc: プログレスバーの説明文

    Returns:
        解凍先のパス
    """
    import subprocess

    if desc is None:
        desc = 'Extracting RAR'

    logger.info(f"Extracting {rar_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    # unrarコマンドを使用して解凍
    # x: 完全パスで解凍, -y: すべてYes
    try:
        result = subprocess.run(
            ['unrar', 'x', '-y', str(rar_path), str(extract_to) + '/'],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Extraction complete: {extract_to}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract RAR: {e.stderr}")
        raise
    except FileNotFoundError:
        raise RuntimeError(
            "unrar command not found. "
            "Please install it: apt-get install unrar (Linux) or brew install unrar (macOS)"
        )

    return extract_to


def extract_archive(archive_path: Path, extract_to: Path, desc: Optional[str] = None) -> Path:
    """
    アーカイブファイルを自動判定して解凍

    Args:
        archive_path: アーカイブファイルのパス
        extract_to: 解凍先ディレクトリ
        desc: プログレスバーの説明文

    Returns:
        解凍先のパス
    """
    suffix = archive_path.suffix.lower()

    if suffix == '.zip':
        return extract_zip(archive_path, extract_to, desc)
    elif suffix in ['.tar', '.gz', '.tgz', '.bz2', '.xz']:
        return extract_tar(archive_path, extract_to, desc)
    elif suffix == '.rar':
        return extract_rar(archive_path, extract_to, desc)
    else:
        raise ValueError(f"Unsupported archive format: {suffix}")


def cleanup_temp_files(temp_dir: Path) -> None:
    """
    一時ファイルをクリーンアップ

    Args:
        temp_dir: 一時ディレクトリのパス
    """
    if temp_dir.exists():
        logger.info(f"Cleaning up temporary files: {temp_dir}")
        shutil.rmtree(temp_dir)
        logger.info("Cleanup complete")


def check_dataset_exists(dataset_path: Path, required_files: Optional[list] = None) -> bool:
    """
    データセットが既に存在するかチェック

    Args:
        dataset_path: データセットのパス
        required_files: 必須ファイルのリスト（パターンマッチ可能）

    Returns:
        存在する場合True
    """
    if not dataset_path.exists():
        return False

    if required_files is None:
        # ディレクトリが存在すればOK
        return True

    # 必須ファイルをチェック
    for pattern in required_files:
        if not list(dataset_path.glob(pattern)):
            logger.warning(f"Required file pattern not found: {pattern}")
            return False

    return True


def move_or_copy_directory(src: Path, dst: Path, move: bool = True) -> None:
    """
    ディレクトリを移動またはコピー

    Args:
        src: 元のパス
        dst: 先のパス
        move: True=移動、False=コピー
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        shutil.rmtree(dst)

    if move:
        shutil.move(str(src), str(dst))
        logger.info(f"Moved {src} -> {dst}")
    else:
        shutil.copytree(src, dst)
        logger.info(f"Copied {src} -> {dst}")
