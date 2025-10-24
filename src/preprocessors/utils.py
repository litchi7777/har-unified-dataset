"""
前処理で使用する共通ユーティリティ関数
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from scipy import signal

logger = logging.getLogger(__name__)


def resample_timeseries(
    data: np.ndarray,
    labels: np.ndarray,
    original_rate: float,
    target_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    時系列データを目標サンプリングレートにリサンプリング

    ポリフェーズフィルタリングを使用して、アンチエイリアスフィルタを
    適用しながら滑らかにリサンプリングします。

    Args:
        data: 入力データ (samples, features)
        labels: ラベル (samples,)
        original_rate: 元のサンプリングレート (Hz)
        target_rate: 目標サンプリングレート (Hz)

    Returns:
        resampled_data: リサンプリングされたデータ
        resampled_labels: リサンプリングされたラベル
    """
    if original_rate == target_rate:
        return data, labels

    num_samples = len(data)
    num_features = data.shape[1]

    # リサンプリング比率を計算（整数で表現）
    # 例: 25Hz -> 30Hz = up=6, down=5 (30/25 = 6/5)
    from math import gcd

    # 比率を簡約化
    rate_ratio = target_rate / original_rate
    # 分数を整数で表現するために、十分大きな数で掛ける
    multiplier = 1000
    up = int(target_rate * multiplier)
    down = int(original_rate * multiplier)

    # 最大公約数で簡約化
    common_divisor = gcd(up, down)
    up = up // common_divisor
    down = down // common_divisor

    logger.info(f"Resampling with polyphase filtering: up={up}, down={down} (ratio={rate_ratio:.4f})")

    # 各チャンネルを個別にリサンプリング（ポリフェーズフィルタリング）
    # 最初のチャンネルでリサンプリングして正確なサイズを取得
    first_channel_resampled = signal.resample_poly(data[:, 0], up, down)
    new_num_samples = len(first_channel_resampled)

    resampled_data = np.zeros((new_num_samples, num_features))
    resampled_data[:, 0] = first_channel_resampled

    for i in range(1, num_features):
        resampled_data[:, i] = signal.resample_poly(data[:, i], up, down)

    # ラベルもリサンプリング（最近傍補間）
    original_indices = np.arange(num_samples)
    new_indices = np.linspace(0, num_samples - 1, new_num_samples)
    resampled_labels = labels[np.round(new_indices).astype(int)]

    logger.info(f"Resampled from {original_rate}Hz to {target_rate}Hz: {data.shape} -> {resampled_data.shape}")

    return resampled_data, resampled_labels


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    stride: int,
    drop_last: bool = False,
    pad_last: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    時系列データにスライディングウィンドウを適用

    Args:
        data: 入力データ (samples, features) or (samples, channels, features)
        labels: ラベル (samples,)
        window_size: ウィンドウサイズ
        stride: ストライド（スライド幅）
        drop_last: 最後の不完全なウィンドウを削除するか
        pad_last: 最後の不完全なウィンドウをパディングするか

    Returns:
        windows: ウィンドウ化されたデータ (num_windows, window_size, ...)
        window_labels: 各ウィンドウのラベル (num_windows,)
    """
    num_samples = len(data)
    windows = []
    window_labels = []

    start = 0
    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        window = data[start:end]
        # ウィンドウ内で最頻のラベルを使用
        window_label_candidates = labels[start:end]
        # 負のラベル（例: -1）を持つサンプルを除外してから最頻値を計算
        valid_labels = window_label_candidates[window_label_candidates >= 0]
        if len(valid_labels) > 0:
            label = np.bincount(valid_labels).argmax()
        else:
            # すべてのラベルが負の場合は-1を使用
            label = -1
        windows.append(window)
        window_labels.append(label)

    # 最後の不完全なウィンドウの処理
    remaining_start = start + stride
    if remaining_start < num_samples:
        remaining_samples = num_samples - remaining_start

        if not drop_last:
            if pad_last and remaining_samples < window_size:
                # パディングして追加
                window = data[remaining_start:]
                pad_width = [(0, window_size - remaining_samples)] + [(0, 0)] * (data.ndim - 1)
                window = np.pad(window, pad_width, mode='edge')  # 最後の値で埋める
                # 負のラベルを除外して最頻値を計算
                remaining_label_candidates = labels[remaining_start:]
                valid_labels = remaining_label_candidates[remaining_label_candidates >= 0]
                if len(valid_labels) > 0:
                    label = np.bincount(valid_labels).argmax()
                else:
                    label = -1
                windows.append(window)
                window_labels.append(label)
            else:
                # 最後のwindow_sizeサンプルを使用
                window = data[-window_size:]
                # 負のラベルを除外して最頻値を計算
                last_label_candidates = labels[-window_size:]
                valid_labels = last_label_candidates[last_label_candidates >= 0]
                if len(valid_labels) > 0:
                    label = np.bincount(valid_labels).argmax()
                else:
                    label = -1
                windows.append(window)
                window_labels.append(label)

    return np.array(windows), np.array(window_labels)


def normalize_data(
    data: np.ndarray,
    method: str = 'standardize',
    axis: Optional[int] = None,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    データを正規化

    Args:
        data: 入力データ
        method: 正規化方法 ('standardize', 'minmax', 'normalize')
        axis: 正規化を適用する軸（Noneの場合は全体）
        epsilon: ゼロ除算を防ぐための小さな値

    Returns:
        正規化されたデータ
    """
    if method == 'standardize':
        # 標準化 (mean=0, std=1)
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + epsilon)

    elif method == 'minmax':
        # Min-Max正規化 [0, 1]
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        return (data - min_val) / (max_val - min_val + epsilon)

    elif method == 'normalize':
        # L2正規化
        norm = np.linalg.norm(data, axis=axis, keepdims=True)
        return data / (norm + epsilon)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def filter_invalid_samples(
    data: np.ndarray,
    labels: np.ndarray,
    threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    無効なサンプル（NaN、Inf、ゼロ分散など）を除去

    Args:
        data: 入力データ
        labels: ラベル
        threshold: 分散の閾値

    Returns:
        フィルタリングされたデータとラベル
    """
    # NaN/Infチェック
    valid_mask = ~(np.isnan(data).any(axis=tuple(range(1, data.ndim))) |
                   np.isinf(data).any(axis=tuple(range(1, data.ndim))))

    # 分散チェック（すべてゼロのサンプルを除外）
    variance = np.var(data, axis=tuple(range(1, data.ndim)))
    valid_mask &= variance > threshold

    filtered_data = data[valid_mask]
    filtered_labels = labels[valid_mask]

    removed_count = len(data) - len(filtered_data)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} invalid samples")

    return filtered_data, filtered_labels


def split_train_val_test(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    データをtrain/val/testに分割

    Args:
        data: 入力データ
        labels: ラベル
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        test_ratio: テストデータの割合
        shuffle: シャッフルするか
        seed: 乱数シード

    Returns:
        train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    num_samples = len(data)
    indices = np.arange(num_samples)

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    return (
        data[train_indices], labels[train_indices],
        data[val_indices], labels[val_indices],
        data[test_indices], labels[test_indices]
    )


def save_npy_dataset(
    output_path: Path,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    test_data: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None
) -> None:
    """
    処理済みデータをNumPy形式で保存

    Args:
        output_path: 出力ディレクトリ
        train_data, train_labels: 訓練データ
        val_data, val_labels: 検証データ（オプション）
        test_data, test_labels: テストデータ（オプション）
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 訓練データ
    np.save(output_path / 'train_data.npy', train_data)
    np.save(output_path / 'train_labels.npy', train_labels)
    logger.info(f"Saved train data: {train_data.shape}")

    # 検証データ
    if val_data is not None and val_labels is not None:
        np.save(output_path / 'val_data.npy', val_data)
        np.save(output_path / 'val_labels.npy', val_labels)
        logger.info(f"Saved val data: {val_data.shape}")

    # テストデータ
    if test_data is not None and test_labels is not None:
        np.save(output_path / 'test_data.npy', test_data)
        np.save(output_path / 'test_labels.npy', test_labels)
        logger.info(f"Saved test data: {test_data.shape}")


def get_class_distribution(labels: np.ndarray) -> dict:
    """
    クラス分布を計算

    Args:
        labels: ラベル配列

    Returns:
        クラスごとのサンプル数
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))
    return distribution
