import pytest
from unittest.mock import patch
import numpy as np
from src.data.data_loader import download_dataset, create_data_generators, print_class_distribution

# Mock kagglehub to avoid actual downloads in CI
@patch('src.data.data_loader.kagglehub.dataset_download')
def test_download_dataset(mock_download):
    mock_download.return_value = "/mock/path"
    path = download_dataset()
    # Expect the function to return config.DATA_DIR, not the mock path directly
    from src.config.config import DATA_DIR
    assert path == DATA_DIR  # Check against the configured DATA_DIR
    mock_download.assert_called_once_with("fanconic/skin-cancer-malignant-vs-benign")

# Skip full test of create_data_generators due to filesystem dependency
def test_create_data_generators_skipped():
    pytest.skip("Requires real filesystem; implement with mock or small dataset for full test")

# Test print_class_distribution with NumPy arrays to match real generator behavior
def test_print_class_distribution(capsys):
    class MockGenerator:
        def __init__(self, classes):
            self.classes = np.array(classes)  # Use NumPy array instead of list

    generators = [
        MockGenerator([0, 1, 0]),  # 2 benign (0), 1 malignant (1)
        MockGenerator([1, 1])      # 0 benign (0), 2 malignant (1)
    ]
    names = ["train", "test"]
    print_class_distribution(generators, names)
    captured = capsys.readouterr()
    assert "Class distribution in the train set" in captured.out
    assert "Benign: 67%" in captured.out  # 2/3 ≈ 67%
    assert "Malignant: 33%" in captured.out  # 1/3 ≈ 33%
    assert "Class distribution in the test set" in captured.out
    assert "Benign: 0%" in captured.out   # 0/2 = 0%
    assert "Malignant: 100%" in captured.out  # 2/2 = 100%