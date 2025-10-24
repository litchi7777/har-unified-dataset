# HAR Unified Dataset

Human Activity Recognition (HAR) データセットの統合前処理・可視化リポジトリ

## 概要

複数のHARデータセットを統一フォーマットで前処理し、可視化するためのツール群です。

## サポートデータセット

- **DSADS** (Daily and Sports Activities Dataset)
- **MHEALTH** (Mobile Health Dataset)
- **OPENPACK** (OpenPack Challenge Dataset)
- **PAMAP2** (Physical Activity Monitoring Dataset)
- **REALWORLD** (Realworld HAR Dataset)

## ディレクトリ構成

```
har-unified-dataset/
├── preprocess.py              # 前処理のエントリーポイント
├── visualize_server.py        # データ可視化Webサーバー
├── src/
│   ├── dataset_info.py        # データセットメタデータ
│   ├── preprocessors/         # データセット別前処理ロジック
│   │   ├── base.py
│   │   ├── common.py
│   │   ├── utils.py
│   │   ├── dsads.py
│   │   ├── mhealth.py
│   │   └── openpack.py
│   └── visualization/         # 可視化ツール
│       └── visualize_data.py
├── configs/                   # 前処理設定ファイル
├── data/
│   ├── raw/                   # 生データ（.gitignoreで除外）
│   └── processed/             # 前処理済みデータ（.gitignoreで除外）
├── outputs/                   # 可視化結果の出力先
└── tests/                     # テストコード

```

## セットアップ

### 依存関係のインストール

```bash
pip install numpy pandas scipy plotly flask pyyaml tqdm requests
```

### データセットのダウンロードと前処理

```bash
# ダウンロードから前処理まで一括実行
python preprocess.py --dataset dsads --download

# 複数のデータセットを処理
python preprocess.py --dataset dsads mhealth openpack --download

# 利用可能なデータセット一覧を表示
python preprocess.py --list
```

## 使い方

### データの前処理

```bash
# 既にダウンロード済みのデータを前処理
python preprocess.py --dataset dsads

# カスタム設定ファイルを使用
python preprocess.py --dataset dsads --config configs/my_config.yaml
```

### データの可視化（Webサーバー）

```bash
# Webサーバーを起動（デフォルト: http://localhost:5000）
python visualize_server.py

# カスタムポートで起動
python visualize_server.py --port 8080
```

ブラウザでアクセスすると、以下の階層的なナビゲーションでデータを探索できます：

1. **トップページ**: データセット一覧
2. **データセットページ**: ユーザー×センサー×モダリティのグリッド
3. **可視化ページ**: 選択したデータのインタラクティブな可視化

## データフォーマット

### 処理済みデータの構造

```
data/processed/{dataset_name}/
├── USER00001/
│   ├── {sensor_name}/
│   │   ├── X.npy      # センサーデータ (n_samples, n_channels, sequence_length)
│   │   └── Y.npy      # ラベル (n_samples,)
├── USER00002/
│   └── ...
```

ユーザー情報はディレクトリ構造で管理されます。

### データセット固有の処理

- **加速度センサーの単位統一**: DSADS、MHEALTHは m/s² → G に変換（scale_factor: 9.8）
- **リサンプリング**: 各データセットを30Hzに統一（OPENPACKは元々30Hzのため不要）
- **データ型最適化**: float16で保存（メモリ効率化）

## テスト

```bash
# 全テスト実行
pytest tests/

# 特定のテスト
pytest tests/test_preprocessing_scale.py
```

## har-foundation との統合

このリポジトリは [har-foundation](https://github.com/litchi7777/har-foundation) のサブモジュールとして使用されます。

```bash
# har-foundation 側での設定
cd /path/to/har-foundation
git submodule add git@github.com:litchi7777/har-unified-dataset.git har-unified-dataset
git submodule update --init --recursive
```

## ライセンス

各データセットのライセンスに準拠してください。

## 参考文献

- DSADS: Barshan, B., & Yüksek, M. C. (2014). Recognizing daily and sports activities...
- MHEALTH: Banos, O., et al. (2014). mHealthDroid: a novel framework for agile development...
- OPENPACK: OpenPack Challenge (https://open-pack.github.io/)
