import pytest
from unittest.mock import patch
from src.data.data_loader import download_dataset, create_data_generators, print_class_distribution

# Mock kagglehub to avoid actual downloads in CI
@patch('src.data.data_loader.kagglehub.dataset_download')
def test_download_dataset(mock_download):
    mock_download.return_value = "/mock/path"
    path = download_dataset()
    assert path == "/mock/path"
    mock_download.assert_called_once_with("fanconic/skin-cancer-malignant-vs-benign")

# Since create_data_generators requires a real directory, we'll skip full testing in CI
# This is a placeholder test; in a real scenario, you'd mock the filesystem or use a small test dataset
def test_create_data_generators_skipped():
    pytest.skip("Requires real filesystem; implement with mock or small dataset for full test")

def test_print_class_distribution(capsys):
    # Mock a generator with classes attribute
    class MockGenerator:
        def __init__(self, classes):
            self.classes = classes

    generators = [MockGenerator([0, 1, 0]), MockGenerator([1, 1])]
    names = ["train", "test"]
    print_class_distribution(generators, names)
    captured = capsys.readouterr()
    assert "Class distribution in the train set" in captured.out
    assert "Benign: 67%" in captured.out
    assert "Malignant: 33%" in captured.out
    assert "Class distribution in the test set" in captured.out
    assert "Benign: 0%" in captured.out
    assert "Malignant: 100%" in captured.out