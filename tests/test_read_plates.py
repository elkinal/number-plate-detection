import pytest
import pandas as pd
import numpy as np
from read_plates import (
    load_data,
    process_plate,
    validate_data,
    generate_summary,
    merge_results,
    calculate_statistics,
)

@pytest.fixture
def large_dataset():
    """Fixture to create a large dataset for performance testing."""
    data = {
        "Well": [f"{chr(65 + i//12)}{i % 12 + 1}" for i in range(96)],  
        "Value": np.random.random(96),
    }
    return pd.DataFrame(data)

def test_large_dataset_performance(large_dataset):
    """Test that large datasets are processed correctly."""
    result = process_plate(large_dataset)
    assert len(result) == len(large_dataset)
    assert "Normalized" in result.columns

def test_invalid_file_path():
    """Test loading from a nonexistent file."""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.csv")

def test_incomplete_plate_data():
    """Test processing when plate data is incomplete."""
    incomplete_data = {
        "Well": ["A1", "A2", "A3"],
        "Value": [0.5, None, 1.2],  
    }
    df = pd.DataFrame(incomplete_data)
    with pytest.raises(ValueError):
        process_plate(df)

def test_merge_results():
    """Test merging multiple plate results."""
    plate1 = pd.DataFrame({"Well": ["A1", "A2"], "Value": [0.1, 0.2]})
    plate2 = pd.DataFrame({"Well": ["A3", "A4"], "Value": [0.3, 0.4]})
    merged = merge_results([plate1, plate2])
    assert len(merged) == 4
    assert all(merged["Well"].isin(["A1", "A2", "A3", "A4"]))

def test_calculate_statistics():
    """Test calculating statistics for a dataset."""
    data = {
        "Value": [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    df = pd.DataFrame(data)
    stats = calculate_statistics(df, "Value")
    assert stats["Mean"] == pytest.approx(0.3)
    assert stats["StdDev"] == pytest.approx(0.158, 0.01)

def test_empty_plate_data():
    """Test processing an empty dataset."""
    empty_df = pd.DataFrame(columns=["Well", "Value"])
    with pytest.raises(ValueError):
        process_plate(empty_df)

def test_malformed_data():
    """Test handling malformed data (unexpected columns)."""
    malformed_data = pd.DataFrame({
        "InvalidColumn1": [1, 2, 3],
        "InvalidColumn2": [4, 5, 6],
    })
    with pytest.raises(KeyError):
        validate_data(malformed_data)

def test_non_numeric_values():
    """Test dataset with non-numeric values in numeric fields."""
    data = {
        "Well": ["A1", "A2", "A3"],
        "Value": [0.1, "Invalid", 0.3], 
    }
    df = pd.DataFrame(data)
    with pytest.raises(ValueError):
        process_plate(df)

def test_generate_summary_edge_cases():
    """Test summary generation for edge cases."""
    # Single row data
    single_row = pd.DataFrame({"Normalized": [0.5]})
    summary = generate_summary(single_row)
    assert summary["Mean"] == pytest.approx(0.5)
    assert summary["StdDev"] == 0.0

    # All zeros
    all_zeros = pd.DataFrame({"Normalized": [0, 0, 0, 0]})
    summary = generate_summary(all_zeros)
    assert summary["Mean"] == 0.0
    assert summary["StdDev"] == 0.0

def test_null_values_in_plate_data():
    """Test dataset with null/NaN values."""
    data_with_nulls = pd.DataFrame({
        "Well": ["A1", "A2", "A3"],
        "Value": [1.0, None, 2.0], 
    })
    with pytest.raises(ValueError):
        process_plate(data_with_nulls)

def test_multiple_plate_merging_edge_case():
    """Test merging results with overlapping well IDs."""
    plate1 = pd.DataFrame({"Well": ["A1", "A2"], "Value": [0.1, 0.2]})
    plate2 = pd.DataFrame({"Well": ["A2", "A3"], "Value": [0.3, 0.4]})
    merged = merge_results([plate1, plate2])
    assert len(merged) == 3
    assert "A2" in merged["Well"].values  

def test_non_standard_well_format():
    """Test handling non-standard well IDs."""
    non_standard_data = pd.DataFrame({
        "Well": ["X1", "X2", "X3"],  
        "Value": [0.5, 0.8, 1.2],
    })
    with pytest.raises(ValueError):
        process_plate(non_standard_data)
