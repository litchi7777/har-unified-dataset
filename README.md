# HAR Unified Dataset

Human Activity Recognition (HAR) データセットの統合前処理・可視化リポジトリ

## 概要

複数のHARデータセットを統一フォーマットで前処理し、可視化するためのツール群です。

**📘 新しいデータセットを追加する場合**: [ADDING_NEW_DATASET.md](ADDING_NEW_DATASET.md)を参照してください。

## サポートデータセット

| データセット | 被験者数 | センサー位置 | センサータイプ | 活動クラス | サンプリングレート | 特徴 |
|------------|---------|------------|--------------|-----------|-----------------|------|
| **DSADS** | 8 | Torso, RightArm, LeftArm, RightLeg, LeftLeg (5箇所) | IMU (ACC, GYRO, MAG) | 19 | 25Hz → 30Hz | 日常・スポーツ活動、全身センサー |
| **MHEALTH** | 10 | Chest, LeftAnkle, RightWrist (3箇所) | IMU (ACC, GYRO, MAG) + ECG | 12 | 50Hz → 30Hz | 健康モニタリング、心電図含む |
| **OPENPACK** | 10 | atr01-04 (4箇所、装着位置は被験者依存) | IMU (ACC, GYRO, QUAT) | 10 | 30Hz | 物流作業、クォータニオン含む |
| **NHANES** | ~13,000 | Waist (1箇所) | ACC | 2 | 80Hz → 30Hz | 大規模健康調査、活動/非活動 |
| **FORTHTRACE** | 15 | LeftWrist, RightWrist, Torso, RightThigh, LeftAnkle (5箇所) | Shimmer IMU (ACC, GYRO, MAG) | 16 | 51.2Hz → 30Hz | 姿勢遷移含む詳細活動認識 |
| **HAR70+** | 18 | LowerBack, RightThigh (2箇所) | Axivity AX3 (ACC) | 7 | 50Hz → 30Hz | 高齢者（70-95歳）特化 |
| **HARTH** | 22 | LowerBack, RightThigh (2箇所) | Axivity AX3 (ACC) | 12 | 50Hz → 30Hz | 自由生活環境、サイクリング含む |
| **REALDISP** | 17 | 全身9箇所（両手足、背中） | IMU (ACC, GYRO, MAG, QUAT) | 33 | 50Hz → 30Hz | センサー配置バリエーション |

## ディレクトリ構成

```
har-unified-dataset/
├── preprocess.py              # 前処理のエントリーポイント
├── visualize_server.py        # データ可視化Webサーバー
├── ADDING_NEW_DATASET.md      # 新規データセット追加ガイド
├── src/
│   ├── dataset_info.py        # データセットメタデータ
│   ├── preprocessors/         # データセット別前処理ロジック
│   │   ├── base.py            # ベースクラス
│   │   ├── common.py          # 共通ユーティリティ（ダウンロード等）
│   │   ├── utils.py           # データ処理ユーティリティ
│   │   ├── dsads.py           # DSADS前処理
│   │   ├── mhealth.py         # MHEALTH前処理
│   │   ├── openpack.py        # OPENPACK前処理
│   │   ├── nhanes_pax.py      # NHANES前処理
│   │   ├── forthtrace.py      # FORTHTRACE前処理
│   │   ├── har70plus.py       # HAR70+前処理
│   │   ├── harth.py           # HARTH前処理
│   │   └── realdisp.py        # REALDISP前処理
│   └── visualization/         # 可視化ツール
│       └── visualize_data.py
├── configs/
│   └── preprocess.yaml        # 前処理設定ファイル
├── data/
│   ├── raw/                   # 生データ（.gitignoreで除外）
│   └── processed/             # 前処理済みデータ（.gitignoreで除外）
├── outputs/                   # 可視化結果の出力先
└── __test__/                  # テストコード

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

統一された階層構造でデータを管理：

```
data/processed/{dataset_name}/
├── USER00001/                    # 1-indexed ユーザーID
│   ├── {sensor_name}/            # センサー位置（例: Chest, LowerBack等）
│   │   ├── {modality}/           # モダリティ（ACC, GYRO, MAG, ECG等）
│   │   │   ├── X.npy             # センサーデータ (n_windows, n_channels, window_size)
│   │   │   └── Y.npy             # ラベル (n_windows,)
│   └── {sensor_name}/
│       └── {modality}/
│           ├── X.npy
│           └── Y.npy
├── USER00002/
│   └── ...
└── metadata.json                 # データセット統計情報
```

**例（FORTHTRACE）:**
```
data/processed/forthtrace/
├── USER00001/
│   ├── LeftWrist/
│   │   ├── ACC/
│   │   │   ├── X.npy  # (N, 3, 150) - 3軸加速度、150サンプル/ウィンドウ
│   │   │   └── Y.npy  # (N,) - 活動ラベル
│   │   ├── GYRO/
│   │   │   └── ...
│   │   └── MAG/
│   │       └── ...
│   ├── RightWrist/
│   └── ...
```

### データセット固有の処理

| データセット | scale_factor | 元レート | 処理後レート | ウィンドウサイズ | 特記事項 |
|------------|-------------|---------|------------|--------------|---------|
| DSADS | 9.8 (m/s²→G) | 25Hz | 30Hz | 150 (5秒) | 全センサー同一モダリティ |
| MHEALTH | 9.8 (m/s²→G) | 50Hz | 30Hz | 150 (5秒) | ECGセンサー含む |
| OPENPACK | なし | 30Hz | 30Hz | 150 (5秒) | クォータニオン（4次元）含む |
| NHANES | なし（G単位） | 80Hz | 30Hz | 150 (5秒) | 単一腰部センサー、大規模 |
| FORTHTRACE | 9.8 (m/s²→G) | 51.2Hz | 30Hz | 150 (5秒) | 姿勢遷移ラベル含む |
| HAR70+ | なし（G単位） | 50Hz | 30Hz | 150 (5秒) | 高齢者特化、加速度のみ |
| HARTH | なし（G単位） | 50Hz | 30Hz | 150 (5秒) | 自由生活環境、サイクリング含む |
| REALDISP | なし（G単位） | 50Hz | 30Hz | 150 (5秒) | クォータニオン含む、3シナリオ |

**共通仕様:**
- **ウィンドウサイズ**: 5秒（全データセット30Hzに統一後、150サンプル）
- **ストライド**: 1秒（80%オーバーラップ、30サンプル）
- **データ型**: float16（メモリ効率化）
- **ユーザーID**: 1-indexed（USER00001から開始）
- **ラベル**: 0-indexed（活動クラス0から開始、未定義クラスは-1）

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

### データセット
- **DSADS**: Barshan, B., & Yüksek, M. C. (2014). Recognizing daily and sports activities in two open source machine learning environments using body-worn sensor units. *The Computer Journal*, 57(11), 1649-1667.
- **MHEALTH**: Banos, O., et al. (2014). mHealthDroid: a novel framework for agile development of mobile health applications. *Ambient Assisted Living and Daily Activities*, 91-98.
- **OPENPACK**: OpenPack Challenge - A Large-scale Benchmark for Activity Recognition in Industrial Settings (https://open-pack.github.io/)
- **NHANES**: National Health and Nutrition Examination Survey (CDC, 2011-2014)
- **FORTHTRACE**: FORTH-TRACE Dataset - Human Activity Recognition with Multi-sensor Data (https://zenodo.org/records/841301)
- **HAR70+**: HAR70+ Dataset - Human Activity Recognition for Older Adults (UCI ML Repository, Dataset #780)
- **HARTH**: HARTH Dataset - Human Activity Recognition Trondheim Dataset (UCI ML Repository, Dataset #779)
- **REALDISP**: Banos, O., Toth, M., & Amft, O. (2012). REALDISP Activity Recognition Dataset. UCI ML Repository (https://doi.org/10.24432/C5GP6D)
