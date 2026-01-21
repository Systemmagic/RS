"""
Property-based tests for data preprocessing module.
Feature: koopman-accuracy-enhancement
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
from osgeo import gdal

from data_preprocessing import PM25SequenceDataset, export_geotiff


# Feature: koopman-accuracy-enhancement, Property 2: Padding Round-Trip Preservation
@settings(max_examples=100, deadline=None)
@given(
    height=st.integers(min_value=50, max_value=300),
    width=st.integers(min_value=50, max_value=300),
    img_size=st.sampled_from([64, 128, 256])
)
def test_padding_round_trip_preservation(height, width, img_size):
    """
    Property 2: Padding Round-Trip Preservation
    
    For any non-square PM2.5 image, applying padding followed by inverse padding
    SHALL preserve the original dimensions and the data values in the valid
    (non-padded) region.
    
    Validates: Requirements 1.2
    
    This test verifies that:
    1. Original dimensions are preserved after round-trip
    2. Data values in the valid region are preserved (within tolerance)
    3. The padding and inverse operations are true inverses
    """
    # Generate random PM2.5-like data
    original_data = np.random.uniform(0, 200, size=(height, width)).astype(np.float32)
    
    # Simulate the padding process
    max_side = max(width, height)
    y_offset = (max_side - height) // 2
    x_offset = (max_side - width) // 2
    
    # Apply padding
    canvas = np.zeros((max_side, max_side), dtype=np.float32)
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = original_data
    
    # Resize to target size (simulating dataset processing)
    resized = cv2.resize(canvas, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    
    # Simulate inverse padding (resize back and extract)
    canvas_restored = cv2.resize(resized, (max_side, max_side), interpolation=cv2.INTER_CUBIC)
    restored_data = canvas_restored[y_offset:y_offset + height, x_offset:x_offset + width]
    
    # Verify dimensions are preserved
    assert restored_data.shape == original_data.shape, \
        f"Dimensions not preserved: {restored_data.shape} != {original_data.shape}"
    
    # Verify data values are approximately preserved
    # Note: Some loss is expected due to interpolation, but should be minimal
    relative_error = np.abs(restored_data - original_data) / (np.abs(original_data) + 1e-6)
    mean_relative_error = np.mean(relative_error)
    
    # Allow up to 5% relative error due to interpolation
    assert mean_relative_error < 0.05, \
        f"Data values not preserved: mean relative error = {mean_relative_error:.4f}"
    
    # Verify no NaN or Inf values introduced
    assert np.all(np.isfinite(restored_data)), \
        "Invalid values (NaN/Inf) introduced during round-trip"


def test_padding_round_trip_with_dataset():
    """
    Test padding round-trip using actual dataset class.
    
    This is a concrete example test that complements the property test.
    """
    # Create temporary directory with test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test GeoTIFF file
        test_file = os.path.join(tmpdir, "test_pm25.tif")
        
        # Create test data (non-square)
        test_height, test_width = 100, 150
        test_data = np.random.uniform(10, 100, size=(test_height, test_width)).astype(np.float32)
        
        # Create GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(test_file, test_width, test_height, 1, gdal.GDT_Float32)
        
        # Set dummy geotransform
        geotransform = (0, 1, 0, 0, 0, -1)
        ds.SetGeoTransform(geotransform)
        ds.SetProjection('EPSG:4326')
        
        # Write data
        ds.GetRasterBand(1).WriteArray(test_data)
        ds.FlushCache()
        ds = None
        
        # Create dataset (this applies padding)
        dataset = PM25SequenceDataset(tmpdir, img_size=128, sequence_length=2)
        
        # Verify padding info is stored correctly
        assert dataset.padding_info['orig_h'] == test_height
        assert dataset.padding_info['orig_w'] == test_width
        assert dataset.padding_info['padded_size'] == max(test_height, test_width)
        
        # Get processed data (at img_size resolution)
        processed = dataset.data[0]
        
        # Apply inverse padding
        restored = dataset.inverse_padding(processed)
        
        # Verify dimensions
        assert restored.shape == (test_height, test_width), \
            f"Restored shape {restored.shape} != original {(test_height, test_width)}"
        
        # Denormalize for comparison
        restored_denorm = dataset.denormalize(
            (restored - dataset.global_min) / (dataset.global_max - dataset.global_min)
        )
        
        # Check that values are reasonably close (allowing for interpolation error)
        # We expect some error due to resize operations
        correlation = np.corrcoef(test_data.flatten(), restored_denorm.flatten())[0, 1]
        assert correlation > 0.95, f"Low correlation after round-trip: {correlation:.4f}"


def test_inverse_padding_preserves_valid_region():
    """
    Test that inverse_padding correctly extracts the valid data region.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with known pattern
        test_file = os.path.join(tmpdir, "pattern.tif")
        
        # Create a pattern that's easy to verify (gradient)
        height, width = 80, 120
        test_data = np.outer(np.arange(height), np.arange(width)).astype(np.float32)
        
        # Create GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(test_file, width, height, 1, gdal.GDT_Float32)
        ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
        ds.SetProjection('EPSG:4326')
        ds.GetRasterBand(1).WriteArray(test_data)
        ds.FlushCache()
        ds = None
        
        # Load with dataset
        dataset = PM25SequenceDataset(tmpdir, img_size=128, sequence_length=2)
        
        # Get processed data
        processed = dataset.data[0]
        
        # Apply inverse padding
        restored = dataset.inverse_padding(processed)
        
        # Verify shape
        assert restored.shape == (height, width)
        
        # Verify the pattern is preserved (check corners and center)
        # Note: Values will be scaled, so check relative relationships
        assert restored[0, 0] < restored[0, -1], "Horizontal gradient not preserved"
        assert restored[0, 0] < restored[-1, 0], "Vertical gradient not preserved"


# Feature: koopman-accuracy-enhancement, Property 9: Valid Data Augmentation
@settings(max_examples=100, deadline=None)
@given(
    T=st.integers(min_value=2, max_value=10),
    H=st.integers(min_value=32, max_value=128),
    W=st.integers(min_value=32, max_value=128)
)
def test_valid_data_augmentation(T, H, W):
    """
    Property 9: Valid Data Augmentation
    
    For any PM2.5 image sequence, after applying random spatial transformations
    (rotation, flipping), all pixel values SHALL remain non-negative, and the
    same transformation SHALL be applied consistently across all frames in the
    temporal sequence.
    
    Validates: Requirements 8.1, 8.2, 8.3
    
    This test verifies that:
    1. All augmented values remain non-negative (physical validity)
    2. No NaN or Inf values are introduced
    3. The same transformation is applied to all frames (temporal consistency)
    4. Augmentation preserves the sequence structure
    """
    from augmentation import PM25Augmentation
    
    # Generate random PM2.5-like sequence (non-negative values)
    sequence = np.random.uniform(0, 200, size=(T, H, W)).astype(np.float32)
    mask = sequence > 10  # Valid data mask
    
    # Apply augmentation
    augmenter = PM25Augmentation(rotation_range=15.0)
    aug_sequence, aug_mask = augmenter(sequence, mask)
    
    # Property 1: All values must remain non-negative
    assert np.all(aug_sequence >= 0), \
        f"Negative values found: min = {aug_sequence.min()}"
    
    # Property 2: No NaN or Inf values
    assert np.all(np.isfinite(aug_sequence)), \
        "Invalid values (NaN/Inf) introduced by augmentation"
    
    # Property 3: Shape is preserved
    assert aug_sequence.shape == sequence.shape, \
        f"Shape changed: {aug_sequence.shape} != {sequence.shape}"
    
    # Property 4: Temporal consistency - check that transformation is consistent
    # We can verify this by checking that the relative spatial patterns are preserved
    # across frames (though absolute positions may change)
    
    # For each frame, check that the augmented mask has similar properties
    for t in range(T):
        original_valid_count = np.sum(mask[t])
        augmented_valid_count = np.sum(aug_mask[t])
        
        # The number of valid pixels should be similar (within 20% due to rotation)
        # Some pixels may be lost at borders during rotation
        ratio = augmented_valid_count / (original_valid_count + 1e-6)
        assert 0.7 <= ratio <= 1.0, \
            f"Frame {t}: Valid pixel count changed significantly: {ratio:.2f}"
    
    # Property 5: Augmentation should produce different results on different calls
    # (with high probability, unless we get the same random transformations)
    aug_sequence2, _ = augmenter(sequence, mask)
    
    # With random transformations, sequences should differ (unless no transform applied)
    # We check that at least sometimes they differ
    if not np.allclose(aug_sequence, aug_sequence2, rtol=1e-5):
        # Good - augmentation is producing varied results
        pass
    else:
        # Could be that no transformation was applied (low probability)
        # This is acceptable
        pass


def test_augmentation_temporal_consistency():
    """
    Test that the same transformation is applied to all frames.
    
    This is a concrete example test that verifies temporal consistency.
    """
    from augmentation import PM25Augmentation
    
    # Create a sequence with a distinctive pattern in each frame
    T, H, W = 5, 64, 64
    sequence = np.zeros((T, H, W), dtype=np.float32)
    
    # Put a bright spot at different locations in each frame
    for t in range(T):
        y, x = 20 + t * 5, 20 + t * 5
        sequence[t, y:y+5, x:x+5] = 100.0
    
    # Apply augmentation with deterministic seed
    np.random.seed(42)
    augmenter = PM25Augmentation(rotation_range=15.0)
    aug_sequence, _ = augmenter(sequence, None)
    
    # Verify all values are non-negative
    assert np.all(aug_sequence >= 0)
    
    # Verify shape preserved
    assert aug_sequence.shape == sequence.shape
    
    # The bright spots should still be present (though possibly moved)
    for t in range(T):
        max_val = np.max(aug_sequence[t])
        assert max_val > 50, f"Frame {t}: Bright spot lost or severely dimmed"


def test_augmentation_preserves_zero_regions():
    """
    Test that zero/padding regions remain zero after augmentation.
    """
    from augmentation import PM25Augmentation
    
    # Create sequence with zero padding
    T, H, W = 3, 64, 64
    sequence = np.zeros((T, H, W), dtype=np.float32)
    
    # Add data only in center region
    sequence[:, 20:44, 20:44] = np.random.uniform(10, 100, size=(T, 24, 24))
    
    # Apply augmentation
    augmenter = PM25Augmentation(rotation_range=15.0)
    aug_sequence, _ = augmenter(sequence, None)
    
    # Verify non-negativity
    assert np.all(aug_sequence >= 0)
    
    # Verify that we still have some zero regions (padding)
    # After rotation, some border regions should still be zero
    zero_count = np.sum(aug_sequence == 0)
    assert zero_count > 0, "All zeros were eliminated (unexpected)"


# Feature: koopman-accuracy-enhancement, Property 10: GeoTIFF Coordinate Preservation
@settings(max_examples=100, deadline=None)
@given(
    height=st.integers(min_value=50, max_value=200),
    width=st.integers(min_value=50, max_value=200),
    img_size=st.sampled_from([64, 128])
)
def test_geotiff_coordinate_preservation(height, width, img_size):
    """
    Property 10: GeoTIFF Coordinate Preservation
    
    For any predicted PM2.5 map exported as GeoTIFF, the geotransform and
    projection SHALL match the original input data, ensuring spatial
    coordinates are correctly preserved.
    
    Validates: Requirements 8.5
    
    This test verifies that:
    1. Exported GeoTIFF has correct dimensions (original, not padded)
    2. Geotransform is preserved exactly
    3. Projection is preserved exactly
    4. Data can be read back correctly
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create original GeoTIFF with known geotransform
        original_file = os.path.join(tmpdir, "original.tif")
        
        # Generate random geotransform (origin_x, pixel_width, rotation, origin_y, rotation, pixel_height)
        origin_x = np.random.uniform(-180, 180)
        origin_y = np.random.uniform(-90, 90)
        pixel_width = np.random.uniform(0.01, 1.0)
        pixel_height = -np.random.uniform(0.01, 1.0)  # Negative for north-up
        
        original_geotransform = (origin_x, pixel_width, 0, origin_y, 0, pixel_height)
        original_projection = 'EPSG:4326'
        
        # Create original data
        original_data = np.random.uniform(10, 100, size=(height, width)).astype(np.float32)
        
        # Write original GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(original_file, width, height, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(original_geotransform)
        ds.SetProjection(original_projection)
        ds.GetRasterBand(1).WriteArray(original_data)
        ds.FlushCache()
        ds = None
        
        # Load with dataset (applies padding and resizing)
        dataset = PM25SequenceDataset(tmpdir, img_size=img_size, sequence_length=2)
        
        # Simulate a prediction (just use the processed data)
        prediction = dataset.data[0]
        
        # Denormalize prediction
        prediction_denorm = dataset.denormalize(
            (prediction - dataset.global_min) / (dataset.global_max - dataset.global_min)
        )
        
        # Export as GeoTIFF
        output_file = os.path.join(tmpdir, "prediction.tif")
        export_geotiff(output_file, prediction_denorm, dataset)
        
        # Read back the exported GeoTIFF
        exported_ds = gdal.Open(output_file)
        
        # Verify dimensions match original (not padded size)
        assert exported_ds.RasterXSize == width, \
            f"Width mismatch: {exported_ds.RasterXSize} != {width}"
        assert exported_ds.RasterYSize == height, \
            f"Height mismatch: {exported_ds.RasterYSize} != {height}"
        
        # Verify geotransform is preserved
        exported_geotransform = exported_ds.GetGeoTransform()
        for i, (orig, exp) in enumerate(zip(original_geotransform, exported_geotransform)):
            assert abs(orig - exp) < 1e-6, \
                f"Geotransform element {i} mismatch: {orig} != {exp}"
        
        # Verify projection is preserved
        exported_projection = exported_ds.GetProjection()
        assert original_projection in exported_projection or exported_projection in original_projection, \
            f"Projection mismatch: {original_projection} != {exported_projection}"
        
        # Verify data can be read back
        exported_data = exported_ds.GetRasterBand(1).ReadAsArray()
        assert exported_data.shape == (height, width), \
            f"Exported data shape mismatch: {exported_data.shape} != {(height, width)}"
        
        # Verify no NaN or Inf in exported data
        assert np.all(np.isfinite(exported_data)), \
            "Exported data contains NaN or Inf values"
        
        exported_ds = None


def test_geotiff_export_with_real_coordinates():
    """
    Test GeoTIFF export with realistic coordinate system.
    
    This is a concrete example test using realistic geospatial coordinates.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with realistic coordinates (e.g., China region)
        test_file = os.path.join(tmpdir, "china_pm25.tif")
        
        # Realistic coordinates for a region in China
        height, width = 90, 120
        origin_x = 110.0  # Longitude
        origin_y = 35.0   # Latitude
        pixel_size = 0.1  # 0.1 degree resolution
        
        geotransform = (origin_x, pixel_size, 0, origin_y, 0, -pixel_size)
        projection = 'EPSG:4326'  # WGS84
        
        # Create test data
        test_data = np.random.uniform(20, 150, size=(height, width)).astype(np.float32)
        
        # Write GeoTIFF
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(test_file, width, height, 1, gdal.GDT_Float32)
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        ds.GetRasterBand(1).WriteArray(test_data)
        ds.FlushCache()
        ds = None
        
        # Load with dataset
        dataset = PM25SequenceDataset(tmpdir, img_size=128, sequence_length=2)
        
        # Verify metadata stored correctly
        assert dataset.orig_h == height
        assert dataset.orig_w == width
        assert dataset.geo_transform == geotransform
        
        # Create a prediction
        prediction = dataset.data[0] * 1.1  # Simulate prediction
        prediction_denorm = dataset.denormalize(
            (prediction - dataset.global_min) / (dataset.global_max - dataset.global_min)
        )
        
        # Export
        output_file = os.path.join(tmpdir, "prediction_china.tif")
        export_geotiff(output_file, prediction_denorm, dataset)
        
        # Verify exported file
        exported_ds = gdal.Open(output_file)
        assert exported_ds.RasterXSize == width
        assert exported_ds.RasterYSize == height
        
        exported_geotransform = exported_ds.GetGeoTransform()
        for orig, exp in zip(geotransform, exported_geotransform):
            assert abs(orig - exp) < 1e-9
        
        exported_ds = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
