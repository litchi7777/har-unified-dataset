"""
Simple test to verify REALDISP preprocessor can be instantiated
"""

import yaml
from src.preprocessors import get_preprocessor

def test_realdisp_instantiation():
    """Test that REALDISP preprocessor can be instantiated"""

    # Load config
    with open('configs/preprocess.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get global config
    global_config = config.get('global', {})
    dataset_config = config.get('datasets', {}).get('realdisp', {})
    dataset_config = {**global_config, **dataset_config}

    # Get preprocessor class
    PreprocessorClass = get_preprocessor('realdisp')

    # Instantiate preprocessor
    preprocessor = PreprocessorClass(dataset_config)

    # Check basic properties
    assert preprocessor.dataset_name == 'realdisp'
    assert preprocessor.num_activities == 33
    assert preprocessor.num_subjects == 17
    assert preprocessor.num_sensors == 9
    assert len(preprocessor.sensor_names) == 9
    assert preprocessor.window_size == 150
    assert preprocessor.stride == 30
    assert preprocessor.target_sampling_rate == 30

    print("✓ REALDISP preprocessor instantiation test passed")
    print(f"  - Dataset: {preprocessor.dataset_name}")
    print(f"  - Activities: {preprocessor.num_activities}")
    print(f"  - Subjects: {preprocessor.num_subjects}")
    print(f"  - Sensors: {preprocessor.num_sensors}")
    print(f"  - Sensor names: {preprocessor.sensor_names}")
    print(f"  - Window size: {preprocessor.window_size}")
    print(f"  - Stride: {preprocessor.stride}")
    print(f"  - Target sampling rate: {preprocessor.target_sampling_rate} Hz")

if __name__ == '__main__':
    test_realdisp_instantiation()
