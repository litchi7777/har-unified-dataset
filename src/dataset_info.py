"""
データセット情報の管理

各HARデータセットのメタデータを定義します。
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


# データセットメタデータ
DATASETS = {
    "DSADS": {
        "sensor_list": ["Torso", "RightArm", "LeftArm", "RightLeg", "LeftLeg"],
        "modalities": ["ACC", "GYRO", "MAG"],
        "n_classes": 19,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 25,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G に変換（加速度のみ）
        "has_undefined_class": False,  # ラベル-1は存在しない（全サンプルが定義済みクラス）
        "labels": {
            0: 'Sitting', 1: 'Standing', 2: 'Lying(Back)', 3: 'Lying(Right)',
            4: 'StairsUp', 5: 'StairsDown', 6: 'Standing(Elevator, still)',
            7: 'Moving(elevator)', 8: 'Walking(parking)',
            9: 'Walking(Treadmill, Flat)', 10: 'Walking(Treadmill, Slope)',
            11: 'Running(treadmill)', 12: 'Exercising(Stepper)',
            13: 'Exercising(Cross trainer)', 14: 'Cycling(Exercise bike, Vertical)',
            15: 'Cycling(Exercise bike, Horizontal)', 16: 'Rowing',
            17: 'Jumping', 18: 'PlayingBasketball'
        },
    },
    "PAMAP2": {
        "sensor_list": ["hand", "chest", "ankle"],
        "n_classes": 12,
        "labels": {
            0: 'lying', 1: 'sitting', 2: 'standing', 3: 'walking',
            4: 'running', 5: 'cycling', 6: 'Nordic walking',
            7: 'descending stairs', 8: 'vacuum clearning',
            9: 'house clearning', 10: 'playing soccer', 11: 'rope jumping'
        },
    },
    "REALWORLD": {
        "sensor_list": ["chest", "forearm", "thigh", "head", "shin", "upperarm", "waist"],
        "n_classes": 8,
        "labels": {
            0: 'Walking', 1: 'Running', 2: 'Sitting', 3: 'Standing',
            4: 'Lying', 5: 'Stairs up', 6: 'Stairs down', 7: 'Jumping'
        },
    },
    "MHEALTH": {
        "sensor_list": ["Chest", "LeftAnkle", "RightWrist"],
        "modalities": {
            "Chest": ["ACC", "ECG"],
            "LeftAnkle": ["ACC", "GYRO", "MAG"],
            "RightWrist": ["ACC", "GYRO", "MAG"]
        },
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G に変換（加速度のみ）
        "has_undefined_class": True,  # ラベル-1が存在
        "labels": {
            -1: 'Undefined',  # 未定義/無活動
            0: 'Standing', 1: 'Sitting', 2: 'LyingDown', 3: 'Walking',
            4: 'StairsUp', 5: 'WaistBendsForward', 6: 'FrontalElevationArms',
            7: 'KneesBending', 8: 'Cycling', 9: 'Jogging',
            10: 'Running', 11: 'JumpFrontBack'
        },
    },
    "OPENPACK": {
        "sensor_list": ["atr01", "atr02", "atr03", "atr04"],
        "modalities": {
            "atr01": ["ACC", "GYRO", "QUAT"],
            "atr02": ["ACC", "GYRO", "QUAT"],
            "atr03": ["ACC", "GYRO", "QUAT"],
            "atr04": ["ACC", "GYRO", "QUAT"]
        },
        "n_classes": 10,
        "sampling_rate": 30,  # Hz
        "original_sampling_rate": 30,  # Hz (リサンプリング不要)
        "has_undefined_class": True,  # ラベル-1が存在
        "labels": {
            -1: 'Undefined',  # 未定義/無操作
            0: 'Assemble', 1: 'Insert', 2: 'Put', 3: 'Walk',
            4: 'Pick', 5: 'Scan', 6: 'Press', 7: 'Open',
            8: 'Close', 9: 'Other'
        },
    },
}


def get_available_sensors(data_root: str) -> List[str]:
    """
    データセットで利用可能なセンサー部位を取得

    Args:
        data_root: データルートパス

    Returns:
        利用可能なセンサー部位のリスト
    """
    data_path = Path(data_root)

    if not data_path.exists():
        raise ValueError(f"Data root not found: {data_root}")

    # サブディレクトリを検索
    sensors = []
    for item in data_path.iterdir():
        if item.is_dir():
            # X.npy, Y.npy, U.npy が存在するかチェック
            required_files = ['X.npy', 'Y.npy', 'U.npy']
            if all((item / f).exists() for f in required_files):
                sensors.append(item.name)

    return sorted(sensors)


def get_dataset_info(dataset_name: str, data_root: str = None) -> Dict[str, any]:
    """
    データセットの基本情報を取得

    Args:
        dataset_name: データセット名（例: "DSADS"）
        data_root: データルートパス（実際のデータから情報を取得する場合）

    Returns:
        データセット情報の辞書
    """
    # メタデータから情報を取得
    if dataset_name in DATASETS:
        meta = DATASETS[dataset_name]
        info = {
            'dataset_name': dataset_name,
            'sensor_list': meta['sensor_list'],
            'n_classes': meta['n_classes'],
            'labels': meta['labels'],
        }

        # data_rootが指定されている場合、実データから追加情報を取得
        if data_root:
            available_sensors = get_available_sensors(data_root)
            info['available_sensors'] = available_sensors

            # 最初のセンサーからユーザー情報とデータサイズを取得
            if available_sensors:
                sensor_path = Path(data_root) / available_sensors[0]
                Y = np.load(sensor_path / "Y.npy")
                U = np.load(sensor_path / "U.npy")
                X = np.load(sensor_path / "X.npy")

                info['num_users'] = len(np.unique(U))
                info['user_ids'] = sorted(np.unique(U).tolist())
                info['num_samples'] = len(Y)
                info['channels_per_sensor'] = X.shape[1] if len(X.shape) > 1 else 1
                info['sequence_length'] = X.shape[2] if len(X.shape) > 2 else X.shape[1]

        return info
    else:
        # データセットが定義されていない場合は、実データから検出
        if not data_root:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in metadata. "
                f"Provide 'data_root' to auto-detect."
            )

        available_sensors = get_available_sensors(data_root)
        if not available_sensors:
            raise ValueError(f"No valid sensor data found in {data_root}")

        # 最初のセンサーから情報を取得
        sensor_path = Path(data_root) / available_sensors[0]
        Y = np.load(sensor_path / "Y.npy")
        U = np.load(sensor_path / "U.npy")
        X = np.load(sensor_path / "X.npy")

        return {
            'dataset_name': dataset_name,
            'available_sensors': available_sensors,
            'sensor_list': available_sensors,
            'num_users': len(np.unique(U)),
            'user_ids': sorted(np.unique(U).tolist()),
            'num_samples': len(Y),
            'n_classes': len(np.unique(Y)),
            'channels_per_sensor': X.shape[1] if len(X.shape) > 1 else 1,
            'sequence_length': X.shape[2] if len(X.shape) > 2 else X.shape[1],
        }


def select_sensors(
    dataset_name: str,
    data_root: str,
    mode: str,
    specific_sensors: Optional[List[str]] = None
) -> List[str]:
    """
    使用するセンサー部位を選択

    Args:
        dataset_name: データセット名
        data_root: データルートパス
        mode: "single_device" or "multi_device"
        specific_sensors: 指定するセンサー部位のリスト（オプション）

    Returns:
        使用するセンサー部位のリスト
    """
    available_sensors = get_available_sensors(data_root)

    if specific_sensors:
        # 指定されたセンサーが利用可能か確認
        for s in specific_sensors:
            if s not in available_sensors:
                raise ValueError(
                    f"Sensor '{s}' not available. "
                    f"Available sensors: {available_sensors}"
                )
        return specific_sensors

    if mode == "single_device":
        # メタデータが存在する場合は最初のセンサーを使用
        if dataset_name in DATASETS:
            first_sensor = DATASETS[dataset_name]['sensor_list'][0]
            if first_sensor in available_sensors:
                return [first_sensor]
        # それ以外は利用可能なセンサーの最初のものを使用
        return [available_sensors[0]]

    elif mode == "multi_device":
        # メタデータが存在する場合は定義されたセンサーリストを使用
        if dataset_name in DATASETS:
            meta_sensors = DATASETS[dataset_name]['sensor_list']
            # 利用可能なセンサーのみをフィルタ
            return [s for s in meta_sensors if s in available_sensors]
        # それ以外は全センサーを使用
        return available_sensors

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'single_device' or 'multi_device'")
