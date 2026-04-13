import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from ekgds.ptbxl import PtbxlProcessingDS


def test_missing_root_dir(monkeypatch):
    """Test that the dataset raises a ValueError if no root folder is provided."""
    # Ensure the environment variable is not set during this test
    monkeypatch.delenv("PTBXL_DATA_DIR", raising=False)

    with pytest.raises(ValueError, match="Dataset root folder must be specified"):
        PtbxlProcessingDS(root_folder=None)


@patch("ekgds.ptbxl.load_single_ptbxl_record")
def test_dataset_getitem(mock_load, tmp_path):
    """Test the dataset loading and preprocessing logic using dummy files."""
    # 1. Create dummy metadata CSVs in the temporary pytest directory
    db_path = tmp_path / "ptbxl_database.csv"
    scp_path = tmp_path / "scp_statements.csv"

    pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [101, 102],
            "scp_codes": ["{'NORM': 100.0}", "{'MI': 100.0}"],
        }
    ).to_csv(db_path, index=False)

    pd.DataFrame(
        {
            "diagnostic_class": ["NORM", "MI"],
            "diagnostic": [1, 1],
            "description": ["Normal ECG", "Myocardial Infarction"],
        },
        index=["NORM", "MI"],
    ).to_csv(scp_path)

    # 2. Mock the wfdb load function to return a dummy signal
    # PTB-XL lowres signals are typically 1000 samples by 12 leads
    mock_sig = np.random.randn(1000, 12)
    mock_meta = {
        "fs": 100,
        "units": ["mV"] * 12,
        "sig_name": [
            "I",
            "II",
            "III",
            "aVR",
            "aVL",
            "aVF",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
        ],
        "comments": ["test comment"],
    }
    mock_load.return_value = (mock_sig, mock_meta)

    # 3. Instantiate the dataset pointing to our temporary directory
    ds = PtbxlProcessingDS(root_folder=str(tmp_path), lowres=True)

    # 4. Assertions
    assert len(ds) == 2

    sig, meta = ds[0]

    # Check that np.apply_along_axis transposes and returns the expected shape (12 leads, 1000 samples)
    assert sig.shape == (12, 1000)

    # Check that metadata was correctly extracted and updated
    assert meta["ecg_id"] == 1
    assert meta["units"] == "mV"
    assert meta["comments"] == "test comment"
