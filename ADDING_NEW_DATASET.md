# 新しいデータセットの追加ガイド

このドキュメントは、har-unified-datasetに新しいHARデータセットを追加する際の手順と心得をまとめたものです。

**対象**: 人間の開発者、AI（Claude Code等）

**重要**: AIがこのガイドを使用する場合、各ステップを順番に実行し、既存の実装（dsads.py、mhealth.py）を**必ず参照**してから実装してください。推測や創作は避け、**パターンに忠実**に従ってください。

---

## 目次

1. [事前準備：データセット理解](#事前準備データセット理解)
2. [データセット追加の5ステップ](#データセット追加の5ステップ)
3. [実装の詳細](#実装の詳細)
4. [重要な設計原則](#重要な設計原則)
5. [チェックリスト](#チェックリスト)
6. [AI実装時の注意事項](#ai実装時の注意事項)

---

## 事前準備：データセット理解

新しいデータセットを追加する前に、以下の情報を完全に理解する必要があります：

### 必須情報

- **データ構造**
  - ファイル形式（CSV, TXT, MAT, NPYなど）
  - ディレクトリ構成（被験者別、活動別など）
  - データの保存形態（連続データ、セグメント単位など）

- **センサー情報**
  - センサー数と装着位置（例：胸部、手首、足首）
  - センサータイプとチャンネル数（ACC: 3軸、GYRO: 3軸、ECG: 2チャンネルなど）
  - 各センサーのモダリティ構成

- **サンプリング情報**
  - オリジナルのサンプリングレート（Hz）
  - サンプル数、セグメント長
  - 時間的な連続性の有無

- **ラベル情報**
  - 活動クラス数
  - ラベルの形式（数値、文字列、階層的など）
  - 未定義クラス（-1やNullなど）の有無
  - ラベルのインデックス方式（0-indexed、1-indexed）

- **データ単位とスケーリング**
  - **最重要**: 加速度センサーの単位（G、m/s²、mg など）
  - ジャイロの単位（rad/s、deg/s など）
  - 正規化やスケーリングが既に適用されているか

- **被験者情報**
  - 被験者数
  - ID形式（連番、文字列など）

---

## データセット追加の5ステップ

### ステップ1: データセットメタデータの登録

**ファイル**: `src/dataset_info.py`

`DATASETS`辞書に新しいデータセットの情報を追加します。

```python
DATASETS = {
    # ...既存のデータセット...

    "YOUR_DATASET": {
        "sensor_list": ["Sensor1", "Sensor2"],  # センサー位置のリスト
        "modalities": ["ACC", "GYRO"],          # 全センサー共通の場合
        # または
        "modalities": {                          # センサーごとに異なる場合
            "Sensor1": ["ACC", "GYRO"],
            "Sensor2": ["ACC", "ECG"]
        },
        "n_classes": 10,                         # 有効なクラス数
        "sampling_rate": 30,                     # リサンプリング後のレート
        "original_sampling_rate": 50,            # オリジナルのレート
        "scale_factor": 9.8,                     # 加速度の単位変換係数（m/s² → G）
        "has_undefined_class": True,             # -1ラベルの有無
        "labels": {
            -1: 'Undefined',  # has_undefined_class=Trueの場合のみ
            0: 'Activity1',
            1: 'Activity2',
            # ...
        },
    },
}
```

#### 重要なポイント

1. **`scale_factor`の決定**（最重要）
   - データセットの加速度が**m/s²単位**の場合：`scale_factor: 9.8`
   - データセットの加速度が**G単位**の場合：`scale_factor`を省略またはNone
   - この係数は**ACCモダリティのみ**に適用される（GYRO、MAGには適用されない）
   - 目的：異なるデータセット間での加速度スケールを**G単位に統一**

2. **`modalities`の指定方法**
   - 全センサーが同じモダリティ構成：リスト形式
   - センサーごとに異なる：辞書形式（MHEALTHの例を参照）

3. **`has_undefined_class`フラグ**
   - ラベル-1（未定義/無活動）が存在する場合：`True`
   - すべてのサンプルが有効なクラス：`False`

---

### ステップ2: 前処理クラスの実装

**ファイル**: `src/preprocessors/your_dataset.py`

`BasePreprocessor`を継承した新しいクラスを作成します。

```python
"""
YOUR_DATASET (Your Dataset Name) 前処理

データセット概要:
- N種類の活動
- M人の被験者
- K個のセンサー
- サンプリングレート: XXHz
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
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

# データセットのURL（公開されている場合）
YOUR_DATASET_URL = "https://example.com/dataset.zip"


@register_preprocessor('your_dataset')
class YourDatasetPreprocessor(BasePreprocessor):
    """
    YOUR_DATASETデータセット用の前処理クラス
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # データセット固有の設定
        self.num_activities = 10
        self.num_subjects = 20
        self.num_sensors = 3

        # センサーとチャンネルのマッピング
        self.sensor_names = ['Sensor1', 'Sensor2', 'Sensor3']
        self.sensor_channel_ranges = {
            'Sensor1': (0, 6),   # channels 0-5
            'Sensor2': (6, 12),  # channels 6-11
            'Sensor3': (12, 18)  # channels 12-17
        }

        # モダリティ（各センサー内のチャンネル分割）
        self.sensor_modalities = {
            'Sensor1': {
                'ACC': (0, 3),   # 3軸加速度
                'GYRO': (3, 6),  # 3軸ジャイロ
            },
            # ...他のセンサー...
        }

        # サンプリングレート
        self.original_sampling_rate = 50  # Hz (オリジナル)
        self.target_sampling_rate = config.get('target_sampling_rate', 30)  # Hz (目標)

        # 前処理パラメータ
        self.window_size = config.get('window_size', 150)  # 5秒 @ 30Hz
        self.stride = config.get('stride', 30)  # 1秒 @ 30Hz

        # スケーリング係数（dataset_info.pyから取得）
        self.scale_factor = DATASETS.get('YOUR_DATASET', {}).get('scale_factor', None)

    def get_dataset_name(self) -> str:
        return 'your_dataset'

    def download_dataset(self) -> None:
        """
        データセットをダウンロードして解凍

        手動ダウンロードが必要な場合は NotImplementedError を発生させる
        """
        # 実装例はdsads.pyを参照

    def load_raw_data(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        生データを被験者ごとに読み込む

        Returns:
            {person_id: (data, labels)} の辞書
                data: (num_samples, num_channels) の配列
                labels: (num_samples,) の配列
        """
        # データセット固有の読み込みロジックを実装
        pass

    def clean_data(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        データのクリーニングとリサンプリング
        """
        cleaned = {}
        for person_id, (person_data, labels) in data.items():
            # 無効なサンプルを除去
            cleaned_data, cleaned_labels = filter_invalid_samples(person_data, labels)

            # リサンプリング（必要な場合）
            if self.original_sampling_rate != self.target_sampling_rate:
                resampled_data, resampled_labels = resample_timeseries(
                    cleaned_data,
                    cleaned_labels,
                    self.original_sampling_rate,
                    self.target_sampling_rate
                )
                cleaned[person_id] = (resampled_data, resampled_labels)
                logger.info(f"USER{person_id:05d} cleaned and resampled: {resampled_data.shape}")
            else:
                cleaned[person_id] = (cleaned_data, cleaned_labels)
                logger.info(f"USER{person_id:05d} cleaned: {cleaned_data.shape}")

        return cleaned

    def extract_features(self, data: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, Dict[str, Dict[str, np.ndarray]]]:
        """
        特徴抽出（センサー×モダリティごとのウィンドウ化とスケーリング）

        Returns:
            {person_id: {sensor/modality: {'X': data, 'Y': labels}}}
        """
        processed = {}

        for person_id, (person_data, labels) in data.items():
            logger.info(f"Processing USER{person_id:05d}")
            processed[person_id] = {}

            # 各センサーについて処理
            for sensor_name in self.sensor_names:
                sensor_start_ch, sensor_end_ch = self.sensor_channel_ranges[sensor_name]
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

                # 各モダリティに分割
                modalities = self.sensor_modalities[sensor_name]
                for modality_name, (mod_start_ch, mod_end_ch) in modalities.items():
                    modality_data = windowed_data[:, :, mod_start_ch:mod_end_ch]

                    # スケーリング適用（ACCのみ、scale_factorが定義されている場合）
                    if modality_name == 'ACC' and self.scale_factor is not None:
                        modality_data = modality_data / self.scale_factor
                        logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

                    # 形状変換: (N, T, C) -> (N, C, T)
                    modality_data = np.transpose(modality_data, (0, 2, 1))

                    # float16に変換（メモリ効率化）
                    modality_data = modality_data.astype(np.float16)

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

        保存形式:
            data/processed/your_dataset/USER00001/Sensor1/ACC/X.npy, Y.npy
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

                X = arrays['X']
                Y = arrays['Y']

                np.save(sensor_modality_path / 'X.npy', X)
                np.save(sensor_modality_path / 'Y.npy', Y)

                user_stats['sensor_modalities'][sensor_modality_name] = {
                    'X_shape': X.shape,
                    'Y_shape': Y.shape,
                    'num_windows': len(Y),
                    'class_distribution': get_class_distribution(Y)
                }

                logger.info(f"Saved {user_name}/{sensor_modality_name}: X{X.shape}, Y{Y.shape}")

            total_stats['users'][user_name] = user_stats

        # メタデータ保存（NumPy型をJSON互換に変換）
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
```

---

### ステップ3: 設定ファイルへの追加

**ファイル**: `configs/preprocess.yaml`

```yaml
datasets:
  # ...既存のデータセット...

  your_dataset:
    # データパス
    raw_data_path: data/raw
    processed_data_path: data/processed

    # サンプリングレート
    target_sampling_rate: 30  # Hz (元はXXHzからリサンプリング)

    # ウィンドウパラメータ
    window_size: 150      # 5秒 @ 30Hz
    stride: 30            # 1秒 @ 30Hz (80%オーバーラップ)

    # データセット情報（参考用）
    num_activities: 10
    num_subjects: 20
    num_channels: 18
    original_sampling_rate: 50  # Hz (元データ)
```

---

### ステップ4: テストの作成

**ファイル**: `tests/test_your_dataset.py`

```python
"""
YOUR_DATASET前処理のテスト
"""

import pytest
import numpy as np
from pathlib import Path

from src.preprocessors.your_dataset import YourDatasetPreprocessor


@pytest.fixture
def config():
    """テスト用の設定"""
    return {
        'raw_data_path': 'data/raw',
        'processed_data_path': 'data/processed',
        'target_sampling_rate': 30,
        'window_size': 150,
        'stride': 30,
    }


def test_preprocessor_initialization(config):
    """前処理クラスの初期化テスト"""
    preprocessor = YourDatasetPreprocessor(config)
    assert preprocessor.dataset_name == 'your_dataset'
    assert preprocessor.target_sampling_rate == 30


def test_scale_factor_loading(config):
    """scale_factorがdataset_info.pyから正しく読み込まれるか"""
    preprocessor = YourDatasetPreprocessor(config)
    # データセットがm/s²単位の場合
    assert preprocessor.scale_factor == 9.8  # or None
```

---

### ステップ5: 動作確認

```bash
# 1. データセットが登録されているか確認
python preprocess.py --list

# 2. 前処理を実行（ダウンロード+処理）
python preprocess.py --dataset your_dataset --download

# 3. 処理済みデータの確認
ls -R data/processed/your_dataset/

# 4. メタデータの確認
cat data/processed/your_dataset/metadata.json

# 5. テストの実行
pytest tests/test_your_dataset.py -v
```

---

## 実装の詳細

### データフロー理解

```
生データ
  ↓ load_raw_data()
{person_id: (data, labels)}  # data: (samples, channels), labels: (samples,)
  ↓ clean_data()
{person_id: (cleaned_data, cleaned_labels)}  # 無効サンプル除去、リサンプリング
  ↓ extract_features()
{person_id: {sensor/modality: {'X': windowed_data, 'Y': windowed_labels}}}
  # X: (num_windows, channels, window_size) - float16
  # Y: (num_windows,) - int
  ↓ save_processed_data()
ディスク保存: data/processed/dataset/USER00001/Sensor/Modality/X.npy, Y.npy
```

### 重要な処理ステップ

#### 1. リサンプリング（clean_data内）

```python
# ポリフェーズフィルタリングによる高品質リサンプリング
resampled_data, resampled_labels = resample_timeseries(
    cleaned_data,
    cleaned_labels,
    self.original_sampling_rate,  # 元のレート
    self.target_sampling_rate      # 目標レート（通常30Hz）
)
```

- すべてのデータセットを**30Hz**に統一
- アンチエイリアスフィルタを適用
- ラベルは最近傍補間

#### 2. ウィンドウ化（extract_features内）

```python
windowed_data, windowed_labels = create_sliding_windows(
    sensor_data,           # (samples, channels)
    labels,               # (samples,)
    window_size=150,      # 5秒 @ 30Hz
    stride=30,            # 1秒 @ 30Hz (80%オーバーラップ)
    drop_last=False,      # 最後の不完全なウィンドウも保持
    pad_last=True         # 不足分をパディング
)
# 出力: windowed_data (num_windows, 150, channels)
#       windowed_labels (num_windows,)
```

- 各ウィンドウのラベル：ウィンドウ内の**最頻値**
- 最後のウィンドウ：`edge`モードでパディング

#### 3. スケーリング（extract_features内）

```python
# ACCモダリティのみ、scale_factorが定義されている場合
if modality_name == 'ACC' and self.scale_factor is not None:
    modality_data = modality_data / self.scale_factor
    logger.info(f"Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")
```

**重要**:
- **ACCのみ**に適用（GYRO、MAG、ECGなどには適用しない）
- 目的：異なるデータセット間で加速度を**G単位に統一**
- m/s²データの場合：9.8で除算してG単位に変換

#### 4. 形状変換とデータ型最適化

```python
# (num_windows, window_size, channels) -> (num_windows, channels, window_size)
modality_data = np.transpose(modality_data, (0, 2, 1))

# メモリ効率化のためfloat16に変換
modality_data = modality_data.astype(np.float16)
```

- PyTorchの畳み込み層の入力形式に合わせる：`(batch, channels, time)`
- float16：ストレージとメモリを約50%削減

---

## 重要な設計原則

### 1. 生データ保持の原則

**正規化は行わない** - データは生のセンサー値のまま保存

```python
# ❌ 間違い：標準化を適用
normalized_data = (data - mean) / std

# ✅ 正しい：生データのまま（スケーリングのみ）
if modality_name == 'ACC' and self.scale_factor is not None:
    data = data / self.scale_factor  # 単位統一のみ
```

**理由**:
- データローダーで動的に正規化する（実験ごとに変更可能）
- 前処理での正規化は不可逆で柔軟性を失う
- メタデータに`'normalization': 'none'`を明記

### 2. 単位統一の原則

**加速度は必ずG単位に統一**

```python
# データセットがm/s²の場合
"scale_factor": 9.8  # dataset_info.pyで定義

# preprocessor内で適用
if modality_name == 'ACC' and self.scale_factor is not None:
    modality_data = modality_data / self.scale_factor
```

**理由**:
- 異なるデータセット間での値の範囲を統一
- モデルが複数データセットで学習する際の安定性向上

### 3. ディレクトリ階層の原則

**必ず以下の構造を守る**:

```
data/processed/{dataset_name}/
├── USER00001/
│   ├── {Sensor1}/
│   │   ├── {Modality1}/
│   │   │   ├── X.npy  # (num_windows, channels, window_size) - float16
│   │   │   └── Y.npy  # (num_windows,) - int
│   │   ├── {Modality2}/
│   │   │   ├── X.npy
│   │   │   └── Y.npy
│   ├── {Sensor2}/
│   │   └── ...
├── USER00002/
│   └── ...
└── metadata.json
```

**重要**:
- ユーザーID: `USER00001`形式（5桁ゼロパディング）
- センサー/モダリティ: パス区切り（例：`Torso/ACC`）
- ファイル名: 必ず`X.npy`と`Y.npy`

### 4. ラベル処理の原則

**未定義クラスは-1で統一**

```python
# ラベル変換例（MHEALTHの場合）
# 元のラベル: 0（無活動）、1-12（有効なクラス）
labels = np.where(labels == 0, -1, labels - 1)
# 結果: -1（未定義）、0-11（有効なクラス）
```

**理由**:
- 訓練時に未定義サンプルをフィルタリング可能
- クラス数の一貫性を保つ（n_classes=12の場合、0-11）

### 5. ログ出力の原則

**処理の各段階で詳細なログを出力**

```python
logger.info(f"USER{person_id:05d}: {data.shape}, Labels: {labels.shape}")
logger.info(f"USER{person_id:05d} cleaned and resampled: {resampled_data.shape}")
logger.info(f"  {sensor_modality_key}: X.shape={X.shape}, Y.shape={Y.shape}")
```

**含めるべき情報**:
- データ形状
- 処理ステップの完了
- スケーリング適用の有無

---

## チェックリスト

### 実装前

- [ ] データセットの論文・ドキュメントを熟読
- [ ] 生データのサンプルをダウンロードして確認
- [ ] **加速度センサーの単位を確認**（G、m/s²、mgなど）
- [ ] センサー配置とチャンネル構成を図解
- [ ] サンプリングレートを確認
- [ ] ラベル体系を理解（未定義クラスの有無、インデックス方式）

### 実装中

- [ ] `dataset_info.py`にメタデータを追加
  - [ ] `scale_factor`を正しく設定（m/s²の場合は9.8）
  - [ ] `has_undefined_class`を設定
  - [ ] `labels`辞書を完全に定義
- [ ] `preprocessors/your_dataset.py`を実装
  - [ ] `@register_preprocessor`デコレータを忘れずに
  - [ ] `scale_factor`を`DATASETS`から読み込む
  - [ ] ACCモダリティのみにスケーリングを適用
  - [ ] float16に変換
  - [ ] 形状を`(N, C, T)`に変換
- [ ] `configs/preprocess.yaml`に設定を追加
- [ ] テストコードを作成

### 実装後

- [ ] `python preprocess.py --list`でデータセットが表示される
- [ ] 前処理が正常に完了する
- [ ] `metadata.json`の内容を確認
  - [ ] `normalization: none`になっているか
  - [ ] `scale_factor`が正しく記録されているか
  - [ ] `data_dtype: float16`になっているか
- [ ] 生成されたX.npyの形状を確認：`(N, C, 150)`
- [ ] 生成されたY.npyの形状を確認：`(N,)`
- [ ] クラス分布を確認（極端な不均衡がないか）
- [ ] データの値の範囲を確認
  - [ ] ACCが適切にスケーリングされているか（-10G ~ +10G程度）
  - [ ] NaN/Infが含まれていないか
- [ ] 可視化サーバーで動作確認：`python visualize_server.py`
- [ ] テストがすべてパスする

### ドキュメント

- [ ] README.mdのサポートデータセット一覧を更新
- [ ] データセット固有の注意事項をコメントに記載
- [ ] 実装時の判断（なぜこの方法を選んだか）をドキュメント化

---

## トラブルシューティング

### よくある問題

#### 1. スケーリングが正しく適用されない

**症状**: ACCの値の範囲が他のデータセットと大きく異なる

**原因**:
- `scale_factor`が未設定または間違った値
- スケーリングの条件分岐が正しくない

**解決策**:
```python
# dataset_info.pyで必ず定義
"scale_factor": 9.8  # m/s²の場合

# preprocessor内で必ず適用
if modality_name == 'ACC' and self.scale_factor is not None:
    modality_data = modality_data / self.scale_factor
    logger.info(f"Applied scale_factor to {sensor_name}/{modality_name}")  # ログ確認
```

#### 2. リサンプリング後にサンプル数が合わない

**症状**: リサンプリング後のdataとlabelsの長さが異なる

**原因**: `resample_timeseries`のラベル補間が不適切

**解決策**: `utils.py`の実装を使用（最近傍補間を使用）

#### 3. ウィンドウ化でラベルが-1になる

**症状**: 有効なデータなのにwindowed_labelsが-1

**原因**: ウィンドウ内に-1が最頻値として選ばれた

**解決策**:
- ウィンドウサイズ/ストライドを調整
- または、-1を含むウィンドウを後処理で除外

#### 4. メタデータJSONのシリアライズエラー

**症状**: `TypeError: Object of type int64 is not JSON serializable`

**原因**: NumPy型がJSON非対応

**解決策**: `save_processed_data`の変換関数を使用（dsads.pyを参照）

---

## 参考実装

既存の実装を参考にしてください：

- **シンプルな例**: `src/preprocessors/dsads.py`
  - 全センサーが同じモダリティ構成
  - 標準的なウィンドウ化とスケーリング

- **複雑な例**: `src/preprocessors/mhealth.py`
  - センサーごとに異なるモダリティ
  - 未定義クラス（-1）の処理
  - ラベル変換（0→-1、1-12→0-11）

- **ダウンロード実装**: `src/preprocessors/dsads.py`の`download_dataset()`
  - UCI Machine Learning Repositoryからのダウンロード
  - 解凍とディレクトリ整理

---

## まとめ

新しいデータセットを追加する際の**最重要ポイント**:

1. **加速度の単位を確認し、scale_factorを正しく設定する**
2. **ACCモダリティのみにスケーリングを適用する**
3. **正規化は行わず、生データを保持する**
4. **ディレクトリ構造とファイル名の規約を守る**
5. **未定義クラスは-1で統一する**
6. **float16とデータ形状(N, C, T)を守る**

これらの原則を守ることで、複数のデータセット間での一貫性が保たれ、har-foundationプロジェクトでの学習が円滑に行えます。

疑問点があれば、既存の実装（dsads.py、mhealth.py）を参照し、必要に応じてコードレビューを依頼してください。

---

## AI実装時の注意事項

AIアシスタント（Claude Code等）がこのガイドを使用して新しいデータセットを追加する場合、以下の点に特に注意してください。

### 🤖 必須の実装手順

#### 1. **既存実装を必ず読む**

実装を開始する前に、**必ず以下のファイルを読んでパターンを理解**してください：

```bash
# 必読ファイル
src/dataset_info.py              # メタデータの定義方法
src/preprocessors/base.py        # 基底クラスの構造
src/preprocessors/dsads.py       # シンプルな実装例
src/preprocessors/mhealth.py     # 複雑な実装例
src/preprocessors/utils.py       # 共通ユーティリティ
configs/preprocess.yaml          # 設定ファイルの形式
```

**推測や創作は禁止** - 必ず既存のパターンに従ってください。

#### 2. **コピー&ペーストから始める**

新しいデータセットの実装は、**最も似ている既存実装をコピー**してから修正してください：

- 全センサーが同じモダリティ → `dsads.py`をコピー
- センサーごとにモダリティが異なる → `mhealth.py`をコピー

```bash
# 例: DSADSベースで新しいデータセットを作成
cp src/preprocessors/dsads.py src/preprocessors/your_dataset.py
# その後、データセット固有の部分のみ修正
```

#### 3. **段階的に実装・テストする**

一度にすべてを実装せず、以下の順序で段階的に進めてください：

```python
# ステップ1: メタデータ登録のみ
# dataset_info.pyに追加 → python preprocess.py --list で確認

# ステップ2: クラスの骨組み
# get_dataset_name()のみ実装 → インスタンス化できるか確認

# ステップ3: データ読み込み
# load_raw_data()を実装 → 1ユーザーのデータを読み込めるか確認

# ステップ4: クリーニング
# clean_data()を実装 → リサンプリングが正しく動作するか確認

# ステップ5: 特徴抽出
# extract_features()を実装 → 形状とスケーリングを確認

# ステップ6: 保存
# save_processed_data()を実装 → ファイルが正しく保存されるか確認
```

各ステップで動作確認してから次に進んでください。

#### 4. **絶対に変えてはいけないもの**

以下の要素は**既存実装と完全に一致**させてください：

##### データ形状
```python
# ✅ 必ずこの形状
X.shape = (num_windows, channels, window_size)  # 例: (1000, 3, 150)
Y.shape = (num_windows,)                        # 例: (1000,)

# ❌ これらは間違い
X.shape = (num_windows, window_size, channels)  # 軸の順序が違う
X.shape = (num_windows, channels)                # ウィンドウ化されていない
```

##### データ型
```python
# ✅ 必ずfloat16
X = X.astype(np.float16)

# ❌ これらは間違い
X = X.astype(np.float32)  # メモリ効率が悪い
X = X.astype(np.float64)  # さらに悪い
```

##### ディレクトリ構造
```python
# ✅ 必ずこの構造
data/processed/{dataset_name}/USER00001/{Sensor}/{Modality}/X.npy
                                                           /Y.npy

# ❌ これらは間違い
data/processed/{dataset_name}/user1/sensor1/acc/X.npy     # 命名規則違反
data/processed/{dataset_name}/USER1/Sensor1_ACC/X.npy     # 階層が違う
```

##### ファイル名
```python
# ✅ 必ずこのファイル名
X.npy  # 大文字のX
Y.npy  # 大文字のY

# ❌ これらは間違い
x.npy, y.npy           # 小文字は不可
data.npy, labels.npy   # 別の名前は不可
```

#### 5. **スケーリングの実装チェックポイント**

スケーリングは最も間違いやすい部分です。以下を厳密にチェックしてください：

```python
# チェック1: dataset_info.pyでscale_factorが定義されているか
DATASETS = {
    "YOUR_DATASET": {
        "scale_factor": 9.8,  # m/s²の場合のみ設定
        # ...
    }
}

# チェック2: Preprocessorで読み込んでいるか
self.scale_factor = DATASETS.get('YOUR_DATASET', {}).get('scale_factor', None)

# チェック3: extract_features内で正しく適用されているか
if modality_name == 'ACC' and self.scale_factor is not None:
    modality_data = modality_data / self.scale_factor
    logger.info(f"  Applied scale_factor={self.scale_factor} to {sensor_name}/{modality_name}")

# チェック4: ログにスケーリング適用が記録されているか確認
# ログに "Applied scale_factor=9.8 to Torso/ACC" のような行が出力される
```

#### 6. **よくある間違いと修正方法**

##### 間違い1: モダリティ名のハードコーディング
```python
# ❌ 間違い
if modality_name == 'acc':  # 小文字

# ✅ 正しい
if modality_name == 'ACC':  # 大文字
```

##### 間違い2: スケーリングをすべてのモダリティに適用
```python
# ❌ 間違い
modality_data = modality_data / self.scale_factor  # 条件なし

# ✅ 正しい
if modality_name == 'ACC' and self.scale_factor is not None:
    modality_data = modality_data / self.scale_factor
```

##### 間違い3: 形状変換の忘れ
```python
# ❌ 間違い
# (N, T, C)のまま保存

# ✅ 正しい
modality_data = np.transpose(modality_data, (0, 2, 1))  # (N, C, T)に変換
```

##### 間違い4: float16変換の忘れ
```python
# ❌ 間違い
# float64やfloat32のまま保存

# ✅ 正しい
modality_data = modality_data.astype(np.float16)
```

##### 間違い5: デコレータの忘れ
```python
# ❌ 間違い
class YourDatasetPreprocessor(BasePreprocessor):

# ✅ 正しい
@register_preprocessor('your_dataset')  # これを忘れずに！
class YourDatasetPreprocessor(BasePreprocessor):
```

#### 7. **実装完了後の検証スクリプト**

実装が完了したら、以下のPythonスクリプトで検証してください：

```python
import numpy as np
from pathlib import Path

# データセット名を指定
DATASET_NAME = "your_dataset"
USER_ID = "USER00001"
SENSOR = "Sensor1"
MODALITY = "ACC"

# パス構築
base_path = Path(f"data/processed/{DATASET_NAME}/{USER_ID}/{SENSOR}/{MODALITY}")
X_path = base_path / "X.npy"
Y_path = base_path / "Y.npy"

# ファイル存在チェック
assert X_path.exists(), f"X.npy not found: {X_path}"
assert Y_path.exists(), f"Y.npy not found: {Y_path}"
print("✓ ファイルが存在します")

# データ読み込み
X = np.load(X_path)
Y = np.load(Y_path)

# 形状チェック
assert X.ndim == 3, f"X should be 3D, got {X.ndim}D"
assert Y.ndim == 1, f"Y should be 1D, got {Y.ndim}D"
assert X.shape[0] == Y.shape[0], f"Sample count mismatch: X={X.shape[0]}, Y={Y.shape[0]}"
assert X.shape[2] == 150, f"Window size should be 150, got {X.shape[2]}"
print(f"✓ 形状が正しい: X{X.shape}, Y{Y.shape}")

# データ型チェック
assert X.dtype == np.float16, f"X should be float16, got {X.dtype}"
print(f"✓ データ型が正しい: {X.dtype}")

# 値の範囲チェック（ACC）
if MODALITY == "ACC":
    assert X.min() > -20, f"ACC値が異常に小さい: {X.min()}"
    assert X.max() < 20, f"ACC値が異常に大きい: {X.max()}"
    print(f"✓ ACC値の範囲が妥当: [{X.min():.2f}, {X.max():.2f}]")

# NaN/Infチェック
assert not np.isnan(X).any(), "X contains NaN"
assert not np.isinf(X).any(), "X contains Inf"
print("✓ NaN/Infが含まれていません")

# ラベルチェック
unique_labels = np.unique(Y)
print(f"✓ ユニークなラベル: {unique_labels}")

print("\n=== すべてのチェックをパスしました ===")
```

#### 8. **デバッグのヒント**

問題が発生した場合：

1. **ログを確認**
   ```bash
   tail -n 100 logs/preprocessing/preprocess.log
   ```

2. **中間データを確認**
   ```python
   # load_raw_data()の直後に追加
   print(f"Loaded data shape: {data.shape}")
   print(f"Loaded labels shape: {labels.shape}")
   print(f"Labels unique: {np.unique(labels)}")
   ```

3. **既存実装と比較**
   ```bash
   # DSADSの処理結果と比較
   ls -lh data/processed/dsads/USER00001/Torso/ACC/
   ls -lh data/processed/your_dataset/USER00001/Sensor1/ACC/

   # ファイルサイズが大きく異なる場合、何かがおかしい
   ```

4. **1ユーザーのみで試す**
   ```python
   # load_raw_data()内で1ユーザーのみ処理
   for subject_id in range(1, 2):  # 1ユーザーのみ
       # ...
   ```

### 🎯 実装の成功基準

以下がすべて満たされていれば実装成功です：

- [ ] `python preprocess.py --list`でデータセットが表示される
- [ ] `python preprocess.py --dataset your_dataset`がエラーなく完了する
- [ ] `data/processed/your_dataset/`以下に正しい階層でファイルが保存される
- [ ] すべての`X.npy`が`(N, C, 150)`の形状
- [ ] すべての`Y.npy`が`(N,)`の形状
- [ ] すべてのデータが`float16`型
- [ ] `metadata.json`が生成され、正しい内容が含まれる
- [ ] ACCのスケーリングが正しく適用されている（ログで確認）
- [ ] 検証スクリプトがすべてパスする
- [ ] 可視化サーバーでデータが表示される

### 📝 実装時のコミュニケーション

AIが実装する際は、以下の情報を明示的に報告してください：

1. **実装開始時**
   - どの既存実装をベースにするか
   - データセットの主要な特徴（センサー数、モダリティ、サンプリングレート）
   - scale_factorの値とその理由

2. **実装中**
   - 各ステップの完了報告
   - 既存実装と異なる部分の説明
   - 不明点や判断が必要な箇所

3. **実装完了時**
   - 検証スクリプトの結果
   - 生成されたファイルの統計情報
   - 既知の制限事項や注意点

これにより、人間が実装をレビューしやすくなり、問題の早期発見が可能になります。
