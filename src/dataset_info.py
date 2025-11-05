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
        "modalities": {
            "hand": ["ACC", "GYRO", "MAG"],
            "chest": ["ACC", "GYRO", "MAG"],
            "ankle": ["ACC", "GYRO", "MAG"]
        },
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G に変換（加速度のみ）
        "has_undefined_class": True,  # ラベル0（transient activities）を-1に変換
        "labels": {
            -1: 'other',  # transient activities (original label 0)
            0: 'lying',   # original label 1
            1: 'sitting',  # original label 2
            2: 'standing',  # original label 3
            3: 'walking',  # original label 4
            4: 'running',  # original label 5
            5: 'cycling',  # original label 6
            6: 'Nordic walking',  # original label 7
            7: 'ascending stairs',  # original label 12
            8: 'descending stairs',  # original label 13
            9: 'vacuum cleaning',  # original label 16
            10: 'ironing',  # original label 17
            11: 'rope jumping'  # original label 24
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
    "FORTHTRACE": {
        "sensor_list": ["LeftWrist", "RightWrist", "Torso", "RightThigh", "LeftAnkle"],
        "modalities": ["ACC", "GYRO", "MAG"],
        "n_classes": 16,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 51.2,  # Hz
        "scale_factor": 9.8,  # m/s^2 -> G に変換（加速度のみ）
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Stand', 1: 'Sit', 2: 'Sit and Talk', 3: 'Walk',
            4: 'Walk and Talk', 5: 'Climb Stairs', 6: 'Climb Stairs and Talk',
            7: 'Stand -> Sit', 8: 'Sit -> Stand', 9: 'Stand -> Sit and Talk',
            10: 'Sit and Talk -> Stand', 11: 'Stand -> Walk', 12: 'Walk -> Stand',
            13: 'Stand -> Climb Stairs', 14: 'Climb Stairs -> Walk',
            15: 'Climb Stairs and Talk -> Walk and Talk'
        },
    },
    "HAR70PLUS": {
        "sensor_list": ["LowerBack", "RightThigh"],
        "modalities": ["ACC"],  # 加速度センサーのみ（3軸）
        "n_classes": 7,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": None,  # 既にG単位なので変換不要
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Walking',       # label=1 -> 0
            1: 'Shuffling',     # label=3 -> 1
            2: 'Stairs Up',     # label=4 -> 2
            3: 'Stairs Down',   # label=5 -> 3
            4: 'Standing',      # label=6 -> 4
            5: 'Sitting',       # label=7 -> 5
            6: 'Lying'          # label=8 -> 6
        },
    },
    "HARTH": {
        "sensor_list": ["LowerBack", "RightThigh"],
        "modalities": ["ACC"],  # 加速度センサーのみ（3軸）
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 50,  # Hz
        "scale_factor": None,  # 既にG単位なので変換不要
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Walking',                   # label=1 -> 0
            1: 'Running',                   # label=2 -> 1 (not in sample, but in spec)
            2: 'Shuffling',                 # label=3 -> 2
            3: 'Stairs Up',                 # label=4 -> 3
            4: 'Stairs Down',               # label=5 -> 4
            5: 'Standing',                  # label=6 -> 5
            6: 'Sitting',                   # label=7 -> 6
            7: 'Lying',                     # label=8 -> 7
            8: 'Cycling Seated',            # label=13 -> 8
            9: 'Cycling Standing',          # label=14 -> 9
            10: 'Cycling Seated Inactive',  # label=130 -> 10
            11: 'Cycling Standing Inactive' # label=140 -> 11 (not in sample, but in spec)
        },
    },
    "LARA": {
        "sensor_list": ["LeftArm", "LeftLeg", "Neck", "RightArm", "RightLeg"],
        "modalities": ["ACC", "GYRO"],  # 加速度センサー（3軸） + ジャイロスコープ（3軸）
        "n_classes": 8,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # 既にG単位なので変換不要
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Stationary',
            1: 'GaitCycle',
            2: 'Step',
            3: 'Upwards',
            4: 'Centred',
            5: 'Downwards',
            6: 'TorsoRotation',
            7: 'OtherMotion'
        },
    },
    "REALDISP": {
        "sensor_list": ["LeftCalf", "LeftThigh", "RightCalf", "RightThigh", "Back",
                       "LeftLowerArm", "LeftUpperArm", "RightLowerArm", "RightUpperArm"],
        "modalities": ["ACC", "GYRO", "MAG", "QUAT"],  # 全センサー共通
        "n_classes": 33,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 50,  # Hz (推定値)
        "scale_factor": None,  # 単位要確認（おそらくG単位）
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Walking', 1: 'Jogging', 2: 'Running', 3: 'Jump up',
            4: 'Jump front & back', 5: 'Jump sideways', 6: 'Jump leg/arms open/closed',
            7: 'Jump rope', 8: 'Trunk twist (arms outstretched)', 9: 'Trunk twist (elbows bent)',
            10: 'Waist bends forward', 11: 'Waist rotation',
            12: 'Waist bends (reach foot with opposite hand)', 13: 'Reach heels backwards',
            14: 'Lateral bend (left+right)', 15: 'Lateral bend with arm up (left+right)',
            16: 'Repetitive forward stretching', 17: 'Upper trunk and lower body opposite twist',
            18: 'Lateral elevation of arms', 19: 'Frontal elevation of arms',
            20: 'Frontal hand claps', 21: 'Frontal crossing of arms',
            22: 'Shoulders high-amplitude rotation', 23: 'Shoulders low-amplitude rotation',
            24: 'Arms inner rotation', 25: 'Knees (alternating) to the breast',
            26: 'Heels (alternating) to the backside', 27: 'Knees bending (crouching)',
            28: 'Knees (alternating) bending forward', 29: 'Rotation on the knees',
            30: 'Rowing', 31: 'Elliptical bike', 32: 'Cycling'
        },
    },
    "MEX": {
        "sensor_list": ["Wrist", "Thigh"],
        "modalities": ["ACC"],  # 加速度センサーのみ（3軸）
        "n_classes": 7,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # 既にG単位（±8g）なので変換不要
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Knee-rolling',
            1: 'Bridging',
            2: 'Pelvic tilt',
            3: 'The Clam',
            4: 'Repeated Extension in Lying',
            5: 'Prone punches',
            6: 'Superman'
        },
    },
    "OPPORTUNITY": {
        "sensor_list": ["BACK", "RUA", "RLA", "LUA", "LLA", "L_SHOE", "R_SHOE", "ACC_SENSORS"],  # 全Body-wornセンサー（113ch）
        "modalities": {
            "BACK": ["ACC", "GYRO", "MAG"],
            "RUA": ["ACC", "GYRO", "MAG"],
            "RLA": ["ACC", "GYRO", "MAG"],
            "LUA": ["ACC", "GYRO", "MAG"],
            "LLA": ["ACC", "GYRO", "MAG"],
            "L_SHOE": ["ACC", "GYRO", "MAG"],
            "R_SHOE": ["ACC", "GYRO", "MAG"],
            "ACC_SENSORS": ["ACC"],  # 残り12個の加速度センサー
        },
        "n_classes": 17,  # Mid-level gesturesの有効クラス数
        "sampling_rate": 30,  # Hz（既に30Hzなのでリサンプリング不要）
        "original_sampling_rate": 30,  # Hz
        "scale_factor": 9.8,  # m/s² -> G に変換（加速度のみ）
        "has_undefined_class": True,  # ラベル-1（Null class）が存在
        "labels": {
            -1: 'Null',  # 未定義/無操作
            0: 'Open Door 1',
            1: 'Open Door 2',
            2: 'Close Door 1',
            3: 'Close Door 2',
            4: 'Open Fridge',
            5: 'Close Fridge',
            6: 'Open Dishwasher',
            7: 'Close Dishwasher',
            8: 'Open Drawer 1',
            9: 'Close Drawer 1',
            10: 'Open Drawer 2',
            11: 'Close Drawer 2',
            12: 'Open Drawer 3',
            13: 'Close Drawer 3',
            14: 'Clean Table',
            15: 'Drink from Cup',
            16: 'Toggle Switch'
        },
    },
    "USCHAD": {
        "sensor_list": ["Hip"],
        "modalities": ["ACC", "GYRO"],  # 加速度センサー（3軸） + ジャイロスコープ（3軸）
        "n_classes": 12,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # 既にG単位なので変換不要
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Walking Forward',
            1: 'Walking Left',
            2: 'Walking Right',
            3: 'Walking Upstairs',
            4: 'Walking Downstairs',
            5: 'Running Forward',
            6: 'Jumping Up',
            7: 'Sitting',
            8: 'Standing',
            9: 'Sleeping',
            10: 'Elevator Up',
            11: 'Elevator Down'
        },
    },
    "SELFBACK": {
        "sensor_list": ["Wrist", "Thigh"],
        "modalities": ["ACC"],  # 加速度センサーのみ（3軸）
        "n_classes": 9,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 100,  # Hz
        "scale_factor": None,  # 既にG単位（±8g）なので変換不要
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Walking Downstairs',
            1: 'Walking Upstairs',
            2: 'Walking Slow',
            3: 'Walking Moderate',
            4: 'Walking Fast',
            5: 'Jogging',
            6: 'Sitting',
            7: 'Standing',
            8: 'Lying'
        },
    },
    "PAAL": {
        "sensor_list": ["Wrist"],
        "modalities": ["ACC"],  # 加速度センサーのみ（3軸）
        "n_classes": 24,
        "sampling_rate": 30,  # Hz (リサンプリング後)
        "original_sampling_rate": 32,  # Hz
        "scale_factor": 0.015,  # 整数値からG単位への変換（8ビット分解能: ±2g）
        "has_undefined_class": False,  # すべてのサンプルが定義済みクラス
        "labels": {
            0: 'Blow Nose',
            1: 'Brush Hair',
            2: 'Brush Teeth',
            3: 'Drink Water',
            4: 'Dusting',
            5: 'Eat Meal',
            6: 'Ironing',
            7: 'Open Bottle',
            8: 'Open Box',
            9: 'Phone Call',
            10: 'Put On Jacket',
            11: 'Put On Shoe',
            12: 'Put On Glasses',
            13: 'Salute',
            14: 'Sit Down',
            15: 'Sneeze/Cough',
            16: 'Stand Up',
            17: 'Take Off Jacket',
            18: 'Take Off Shoe',
            19: 'Take Off Glasses',
            20: 'Type On Keyboard',
            21: 'Washing Dishes',
            22: 'Washing Hands',
            23: 'Writing'
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
            # X.npy, Y.npy が存在するかチェック
            required_files = ['X.npy', 'Y.npy']
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

            # 最初のセンサーからデータサイズを取得
            if available_sensors:
                sensor_path = Path(data_root) / available_sensors[0]
                Y = np.load(sensor_path / "Y.npy")
                X = np.load(sensor_path / "X.npy")

                # ユーザー数はディレクトリ構造から取得
                user_dirs = [d.name for d in Path(data_root).parent.iterdir()
                            if d.is_dir() and d.name.startswith('USER')]
                info['num_users'] = len(user_dirs)
                info['user_ids'] = sorted(user_dirs)
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
        X = np.load(sensor_path / "X.npy")

        # ユーザー数はディレクトリ構造から取得
        user_dirs = [d.name for d in Path(data_root).parent.iterdir()
                    if d.is_dir() and d.name.startswith('USER')]

        return {
            'dataset_name': dataset_name,
            'available_sensors': available_sensors,
            'sensor_list': available_sensors,
            'num_users': len(user_dirs),
            'user_ids': sorted(user_dirs),
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
