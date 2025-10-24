"""
前処理のスケーリング機能のテスト
"""
import numpy as np
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_info import DATASETS


def test_dsads_scale_factor_exists():
    """DSADSデータセットにscale_factorが定義されているか確認"""
    assert 'DSADS' in DATASETS
    assert 'scale_factor' in DATASETS['DSADS']
    assert DATASETS['DSADS']['scale_factor'] == 9.8


def test_mhealth_scale_factor_exists():
    """MHEALTHデータセットにscale_factorが定義されているか確認"""
    assert 'MHEALTH' in DATASETS
    assert 'scale_factor' in DATASETS['MHEALTH']
    assert DATASETS['MHEALTH']['scale_factor'] == 9.8


def test_scale_conversion():
    """スケーリングの変換が正しいか確認"""
    # m/s^2からGへの変換
    scale_factor = 9.8

    # 例: 9.8 m/s^2 -> 1.0 G
    data_ms2 = np.array([9.8, 19.6, 4.9])
    expected_g = np.array([1.0, 2.0, 0.5])

    result = data_ms2 / scale_factor
    np.testing.assert_array_almost_equal(result, expected_g, decimal=5)


def test_float16_conversion():
    """float16への変換が正しく行われるか確認"""
    data = np.array([1.234567, 2.345678, 3.456789], dtype=np.float32)
    data_float16 = data.astype(np.float16)

    # float16の精度範囲内で正しく変換されているか
    assert data_float16.dtype == np.float16
    # float16の精度は約3桁なので、大きな差はないことを確認
    np.testing.assert_array_almost_equal(data, data_float16, decimal=2)


if __name__ == '__main__':
    # 全テストを実行
    print("Testing DSADS scale_factor...")
    test_dsads_scale_factor_exists()
    print("✓ DSADS scale_factor test passed")

    print("\nTesting MHEALTH scale_factor...")
    test_mhealth_scale_factor_exists()
    print("✓ MHEALTH scale_factor test passed")

    print("\nTesting scale conversion...")
    test_scale_conversion()
    print("✓ Scale conversion test passed")

    print("\nTesting float16 conversion...")
    test_float16_conversion()
    print("✓ Float16 conversion test passed")

    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)
