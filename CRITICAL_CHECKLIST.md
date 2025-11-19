# 🚨 クリティカルチェックリスト（新規データセット追加時の必須確認事項）

新しいデータセット追加時に、**必ず**以下を確認してください。過去に重大なバグが発生した項目です。

## ✅ 1. 活動ラベルマッピングの検証（最重要）

**問題例**: SBRHAPTデータセットで、activity_labels.txtの順序と実装が完全に不一致

**必須手順**:
```bash
# ステップ1: 元データのラベル定義ファイルを確認
cat data/raw/your_dataset/activity_labels.txt
# または
cat data/raw/your_dataset/README.md

# ステップ2: プリプロセッサの実装を確認
grep -A 20 "activity_labels\|activity_map" src/preprocessors/your_dataset.py

# ステップ3: dataset_info.pyを確認
grep -A 15 "YOUR_DATASET" src/dataset_info.py | grep -A 12 "labels"
```

**チェック項目**:
- [ ] **元データのラベル定義ファイルを必ず読む**（README, activity_labels.txt, labels.csv等）
- [ ] 元データの activity_id と実装の activity_labels/activity_map が完全一致
- [ ] プリプロセッサと dataset_info.py の labels が完全一致
- [ ] 1-indexed → 0-indexed 変換が正しく実装されている（該当する場合）
- [ ] ラベル順序を**推測しない**（必ず元データから確認）

**検証コマンド**:
```python
# 簡単な検証スクリプト
from src.preprocessors import get_preprocessor
from src.dataset_info import DATASETS

config = {'target_sampling_rate': 30, 'window_size': 150, 'stride': 30}
preprocessor = get_preprocessor('your_dataset')(config)

# プリプロセッサのラベル
print("Preprocessor labels:", preprocessor.activity_labels)

# dataset_info.pyのラベル
print("Dataset info labels:", DATASETS['YOUR_DATASET']['labels'])

# 必ず一致していることを確認！
```

---

## ✅ 2. データスケールの検証（最重要）

**問題例**: IMSB、IMWSHAで加速度が異常に大きい（平均7G等）

**必須手順**:
```python
# ステップ1: 生データのスケールを確認
import pandas as pd
import numpy as np

# サンプルデータを読み込み
df = pd.read_csv('data/raw/your_dataset/sample.csv')

# 加速度データの統計を確認
print(df[['acc_x', 'acc_y', 'acc_z']].describe())
print("Min/Max:", df[['acc_x', 'acc_y', 'acc_z']].min().min(),
      df[['acc_x', 'acc_y', 'acc_z']].max().max())
print("Mean:", df[['acc_x', 'acc_y', 'acc_z']].mean())
```

**チェック項目**:
- [ ] **元データのREADME/論文でセンサー単位を確認**
  - G単位: `scale_factor = None`
  - m/s²単位: `scale_factor = 9.8`
  - mg単位: `scale_factor = 0.001 * 9.8`（または1000で除算後に9.8）
- [ ] 加速度の値が合理的な範囲内か（通常±2-4G、激しい動きで±8G程度）
- [ ] 重力成分が含まれているか（静止時に1軸が約±1G）
- [ ] ジャイロの単位（通常rad/s または deg/s）
- [ ] `scale_factor`は**ACCモダリティのみ**に適用される

**正常な値の目安**:
```
加速度（G単位）:
- 静止状態: 重力方向の軸が約±1G、他の軸は約0G
- 歩行: ±2-3G
- ランニング: ±4-6G
- ジャンプ: ±8G程度

ジャイロ（rad/s）:
- 静止状態: 約0 rad/s
- 通常の動作: ±1-3 rad/s
- 速い回転: ±5 rad/s程度
```

**異常値の例**:
```python
# ❌ 異常: 平均が7G（重力方向以外も大きい）
mean: wx=7.4G, wy=7.8G, wz=3.0G
# → スケーリングが間違っている、または単位が不明

# ✅ 正常: 静止時に1軸が約1G
mean: x=0.1G, y=9.8G, z=0.2G  # 元データがm/s²単位
# → scale_factor=9.8 が必要
```

---

## ✅ 3. ファイル名パターンの検証

**問題例**: IMWSHAで「3-imu-one subject.csv」しか探さず、Subject 2以降を無視

**必須手順**:
```bash
# ステップ1: 全ファイルのパターンを確認
find data/raw/your_dataset -name "*.csv" | head -20

# ステップ2: 各被験者のファイル名を確認
for dir in data/raw/your_dataset/Subject*; do
    echo "=== $dir ==="; ls "$dir"/*.csv
done
```

**チェック項目**:
- [ ] 全被験者のファイル名パターンが一致するか確認
- [ ] ハードコードされたファイル名を使用しない
- [ ] `glob('*.csv')` や `glob('3-imu*.csv')` でパターンマッチング
- [ ] 複数ファイル形式がある場合は全パターンに対応

---

## ✅ 4. 被験者数の検証

**必須手順**:
```bash
# 処理後のユーザー数を確認
ls data/processed/your_dataset/ | grep USER | wc -l

# 期待される被験者数と一致するか確認
cat configs/preprocess.yaml | grep -A 5 "your_dataset:" | grep num_subjects
```

**チェック項目**:
- [ ] 処理後のUSER数が、期待される被験者数と一致
- [ ] metadata.jsonのusersキーに全被験者が含まれる
- [ ] ログで「Loaded N subjects successfully」を確認

---

## ✅ 5. クラス数の整合性

**必須手順**:
```python
# dataset_info.pyのn_classes と labels の整合性を確認
from src.dataset_info import DATASETS

dataset = DATASETS['YOUR_DATASET']
n_classes = dataset['n_classes']
labels = dataset['labels']

# Undefinedクラス(-1)を除外した定義済みクラス数
defined_classes = [k for k in labels.keys() if k >= 0]

print(f"n_classes: {n_classes}")
print(f"Defined classes: {len(defined_classes)}")
print(f"Labels: {sorted(defined_classes)}")

# n_classes == len(defined_classes) であることを確認！
assert n_classes == len(defined_classes), "n_classes mismatch!"
```

**チェック項目**:
- [ ] `n_classes` = 定義済みクラス数（Undefined除く）
- [ ] `has_undefined_class=True` の場合、-1ラベルが`labels`に存在
- [ ] ラベルが0から連番であること（0, 1, 2, ..., n_classes-1）

---

## 実装完了後の最終チェック

**全ての新規データセットで以下を実行**:

### 1. プリプロセッサの読み込み確認
```bash
python -c "from src.preprocessors import get_preprocessor; \
    p = get_preprocessor('your_dataset')({'target_sampling_rate': 30}); \
    print('✓ Preprocessor loads successfully')"
```

### 2. ラベルマッピングの検証
```python
from src.preprocessors import get_preprocessor
from src.dataset_info import DATASETS

config = {'target_sampling_rate': 30, 'window_size': 150, 'stride': 30}
preprocessor = get_preprocessor('your_dataset')(config)

# プリプロセッサのラベル
p_labels = preprocessor.activity_labels  # または activity_map
print("Preprocessor labels:", p_labels)

# dataset_info.pyのラベル
d_labels = DATASETS['YOUR_DATASET']['labels']
print("Dataset info labels:", d_labels)

# 一致確認
if hasattr(preprocessor, 'activity_labels'):
    # 1-indexed → 0-indexed マッピングの場合
    for original_id, new_id in p_labels.items():
        assert d_labels[new_id] is not None, f"Label {new_id} not in dataset_info"
    print("✓ Labels match!")
elif hasattr(preprocessor, 'activity_map'):
    # 文字列 → 0-indexed マッピングの場合
    for activity_str, new_id in p_labels.items():
        assert d_labels[new_id] is not None, f"Label {new_id} not in dataset_info"
    print("✓ Labels match!")
```

### 3. スケールの確認（処理後）
```python
import numpy as np
from pathlib import Path

# 処理後のデータを確認
user_dir = Path('data/processed/your_dataset/USER00001')
for sensor_mod in user_dir.rglob('X.npy'):
    X = np.load(sensor_mod)
    print(f"\n{sensor_mod.parent}:")
    print(f"  Shape: {X.shape}")
    print(f"  Min/Max: {X.min():.2f} / {X.max():.2f}")
    print(f"  Mean: {X.mean():.2f}")

    # ACCの場合、合理的な範囲内か確認
    if 'ACC' in str(sensor_mod):
        if abs(X.mean()) > 5.0:
            print("  ⚠️ WARNING: Average acceleration > 5G, check scale_factor!")
        if abs(X.max()) > 15.0:
            print("  ⚠️ WARNING: Max acceleration > 15G, check scale_factor!")
```

---

## まとめ

### 過去の重大バグ

1. **SBRHAPT**: ラベル順序が完全に間違っていた → 全データが誤分類
2. **IMWSHA**: ファイル名パターンが1人分しか対応せず → 9人分のデータが無視
3. **OPENPACK**: `n_classes` の数が間違っていた → クラス数の不整合

### 再発防止のための鉄則

- ✅ **元データのドキュメントを必ず読む**（推測しない）
- ✅ **3つのソースの一致を確認**（元データ、プリプロセッサ、dataset_info.py）
- ✅ **処理後のデータで異常値を確認**（スケール、被験者数、クラス分布）
- ✅ **このチェックリストを必ず実行**（スキップ厳禁）
