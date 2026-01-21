"""
Enhanced data preprocessing module for PM2.5 spatiotemporal forecasting.
Supports high-resolution processing with aspect-ratio-aware padding.
"""

import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from osgeo import gdal
from typing import Tuple, Dict, Optional


class PM25SequenceDataset(Dataset):
    """
    Enhanced PM2.5 sequence dataset with high-resolution support.
    
    Features:
    - Support for 128×128 or 256×256 resolution (default: 128)
    - Bicubic interpolation for better quality
    - Aspect-ratio-aware padding with metadata storage
    - Optional data augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        img_size: int = 128,  # Changed from 64 to 128
        sequence_length: int = 7,
        augment: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing .tif files
            img_size: Target square size (128 or 256)
            sequence_length: Number of consecutive frames in a sequence
            augment: Enable random augmentation (rotation, flipping)
        """
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.augment = augment
        
        # Find all TIFF files
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.tif")))
        
        if not self.file_paths:
            raise FileNotFoundError(f"No .tif files found in {data_dir}")
        
        print(f"Dataset: Found {len(self.file_paths)} files.")
        print(f"Processing with resolution {img_size}×{img_size} and bicubic interpolation...")
        
        # Read metadata from first file
        ref_ds = gdal.Open(self.file_paths[0])
        self.orig_w = ref_ds.RasterXSize
        self.orig_h = ref_ds.RasterYSize
        self.geo_transform = ref_ds.GetGeoTransform()
        self.projection = ref_ds.GetProjection()
        
        # Calculate padding parameters
        max_side = max(self.orig_w, self.orig_h)
        self.y_offset = (max_side - self.orig_h) // 2
        self.x_offset = (max_side - self.orig_w) // 2
        self.padded_size = max_side
        
        # Store padding metadata for later reconstruction
        self.padding_info = {
            'orig_h': self.orig_h,
            'orig_w': self.orig_w,
            'y_offset': self.y_offset,
            'x_offset': self.x_offset,
            'padded_size': self.padded_size
        }
        
        print(f"Original dimensions: {self.orig_h}×{self.orig_w}")
        print(f"Padding to: {max_side}×{max_side}")
        print(f"Offsets: y={self.y_offset}, x={self.x_offset}")
        
        # Load and preprocess all data
        self.data = []
        for p in self.file_paths:
            ds = gdal.Open(p)
            band = ds.GetRasterBand(1)
            arr = band.ReadAsArray()
            
            # Apply padding to make square
            canvas = np.zeros((max_side, max_side), dtype=np.float32)
            canvas[self.y_offset:self.y_offset + self.orig_h,
                   self.x_offset:self.x_offset + self.orig_w] = arr
            
            # Resize using bicubic interpolation (better quality than INTER_AREA)
            arr_resized = cv2.resize(
                canvas,
                (self.img_size, self.img_size),
                interpolation=cv2.INTER_CUBIC
            )
            
            self.data.append(arr_resized)
        
        self.data = np.array(self.data, dtype=np.float32)
        
        # Handle NaN/Inf values
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate global statistics for normalization
        self.global_min = np.min(self.data)
        self.global_max = np.max(self.data)
        
        print(f"Data loaded. Range: [{self.global_min:.2f}, {self.global_max:.2f}]")
        print(f"Shape: {self.data.shape}")
    
    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sequence of PM2.5 maps.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Tuple of (sequence, mask, metadata)
            - sequence: (T, 1, H, W) tensor of normalized PM2.5 values
            - mask: (T, 1, H, W) boolean tensor indicating valid data
            - metadata: Dictionary with padding info and timestamps
        """
        # Extract sequence
        seq = self.data[idx:idx + self.sequence_length]
        
        # Normalize to [0, 1]
        if self.global_max > self.global_min:
            seq = (seq - self.global_min) / (self.global_max - self.global_min)
        
        # Create mask (valid data > 0)
        mask = (seq > 0.0)
        
        # Apply augmentation if enabled
        if self.augment:
            seq, mask = self.apply_augmentation(seq, mask)
        
        # Convert to tensors
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(1)
        
        # Metadata
        metadata = {
            'index': idx,
            'padding_info': self.padding_info,
            'file_paths': self.file_paths[idx:idx + self.sequence_length]
        }
        
        return seq_tensor, mask_tensor, metadata
    
    def apply_augmentation(
        self,
        seq: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random spatial transformations to sequence.
        
        The same transformation is applied to all frames in the sequence
        to maintain temporal consistency.
        
        Args:
            seq: Sequence array (T, H, W)
            mask: Mask array (T, H, W)
            
        Returns:
            Augmented (seq, mask) tuple
        """
        # Random rotation (±15 degrees)
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            center = (seq.shape[2] // 2, seq.shape[1] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply same rotation to all frames
            for t in range(seq.shape[0]):
                seq[t] = cv2.warpAffine(
                    seq[t],
                    rotation_matrix,
                    (seq.shape[2], seq.shape[1]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                mask[t] = cv2.warpAffine(
                    mask[t].astype(np.float32),
                    rotation_matrix,
                    (mask.shape[2], mask.shape[1]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                ) > 0.5
        
        # Random horizontal flip
        if np.random.rand() < 0.5:
            seq = np.flip(seq, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        # Random vertical flip
        if np.random.rand() < 0.5:
            seq = np.flip(seq, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Ensure non-negativity (clip any negative values from rotation)
        seq = np.maximum(seq, 0.0)
        
        return seq, mask
    
    def inverse_padding(self, padded_img: np.ndarray) -> np.ndarray:
        """
        Remove padding to restore original dimensions.
        
        Args:
            padded_img: Image at padded square size (padded_size × padded_size)
            
        Returns:
            Image at original dimensions (orig_h × orig_w)
        """
        # First, resize from img_size back to padded_size
        if padded_img.shape[0] != self.padded_size:
            canvas = cv2.resize(
                padded_img,
                (self.padded_size, self.padded_size),
                interpolation=cv2.INTER_CUBIC
            )
        else:
            canvas = padded_img
        
        # Extract the original region (remove padding)
        actual_img = canvas[
            self.y_offset:self.y_offset + self.orig_h,
            self.x_offset:self.x_offset + self.orig_w
        ]
        
        return actual_img
    
    def get_all_data(self) -> np.ndarray:
        """
        Get all data normalized to [0, 1].
        
        Returns:
            Normalized data array (N, H, W)
        """
        if self.global_max > self.global_min:
            return (self.data - self.global_min) / (self.global_max - self.global_min)
        return self.data
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.
        
        Args:
            normalized_data: Data in [0, 1] range
            
        Returns:
            Data in original scale
        """
        return normalized_data * (self.global_max - self.global_min) + self.global_min


def export_geotiff(
    filename: str,
    data_array: np.ndarray,
    dataset_obj: PM25SequenceDataset
) -> None:
    """
    Export prediction as GeoTIFF with correct coordinates.
    
    Automatically removes padding to restore original dimensions
    and preserves geospatial metadata.
    
    Args:
        filename: Output file path
        data_array: Prediction array (at img_size resolution)
        dataset_obj: Dataset object containing metadata
    """
    # Remove padding to restore original dimensions
    actual_img = dataset_obj.inverse_padding(data_array)
    
    # Create GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        filename,
        dataset_obj.orig_w,
        dataset_obj.orig_h,
        1,
        gdal.GDT_Float32
    )
    
    # Set geospatial metadata
    out_ds.SetGeoTransform(dataset_obj.geo_transform)
    out_ds.SetProjection(dataset_obj.projection)
    
    # Write data
    out_ds.GetRasterBand(1).WriteArray(actual_img)
    out_ds.FlushCache()
    out_ds = None
    
    print(f"Exported GeoTIFF: {filename}")
