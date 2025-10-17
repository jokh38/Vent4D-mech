"""
Validation Tools for Ventilation Analysis

This module provides validation and quality control tools for ventilation calculations.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging


class ValidationTools:
    """
    Validation tools for ventilation analysis.

    This class provides methods for validating ventilation calculations and
    performing quality control checks.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ValidationTools instance.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def validate_ventilation(self, ventilation_map: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform quality control checks on a single ventilation map.

        Args:
            ventilation_map: Ventilation map (D, H, W)
            mask: Optional lung mask to define the region of interest

        Returns:
            Dictionary of validation metrics
        """
        if mask is not None:
            vent_data = ventilation_map[mask]
        else:
            vent_data = ventilation_map.flatten()

        # Remove NaNs and Infs for robust statistics
        vent_data = vent_data[np.isfinite(vent_data)]

        if vent_data.size == 0:
            return {'error': 'No valid data in ventilation map'}

        stats = {
            'min': float(np.min(vent_data)),
            'max': float(np.max(vent_data)),
            'mean': float(np.mean(vent_data)),
            'std': float(np.std(vent_data)),
            'percentile_5': float(np.percentile(vent_data, 5)),
            'percentile_95': float(np.percentile(vent_data, 95)),
            'nan_count': int(np.sum(np.isnan(ventilation_map))),
            'inf_count': int(np.sum(np.isinf(ventilation_map))),
        }
        return stats

    def compare_with_spect(self, ventilation_map: np.ndarray, spect_map: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Compare a calculated ventilation map with a ground truth SPECT map.

        Args:
            ventilation_map: Calculated ventilation map (e.g., from Jacobian)
            spect_map: Ground truth SPECT ventilation map
            mask: Lung mask defining the region for comparison

        Returns:
            Dictionary of comparison metrics
        """
        from scipy.stats import spearmanr

        if ventilation_map.shape != spect_map.shape or ventilation_map.shape != mask.shape:
            raise ValueError("All input maps (ventilation, SPECT, mask) must have the same shape.")

        # Apply mask to get the data within the lungs
        vent_flat = ventilation_map[mask].flatten()
        spect_flat = spect_map[mask].flatten()

        if vent_flat.size == 0:
            return {'error': 'Mask is empty, no data to compare.'}

        # 1. Spearman's Rank Correlation
        spearman_corr, spearman_p_value = spearmanr(vent_flat, spect_flat)

        # 2. Dice Similarity Coefficient for functional regions
        # Define high and low ventilation regions (e.g., based on median)
        vent_median = np.median(vent_flat)
        spect_median = np.median(spect_flat)

        vent_high_mask = (ventilation_map > vent_median) & mask
        spect_high_mask = (spect_map > spect_median) & mask

        vent_low_mask = (ventilation_map <= vent_median) & mask
        spect_low_mask = (spect_map <= spect_median) & mask

        # Calculate Dice
        dice_high = 2 * np.sum(vent_high_mask & spect_high_mask) / (np.sum(vent_high_mask) + np.sum(spect_high_mask))
        dice_low = 2 * np.sum(vent_low_mask & spect_low_mask) / (np.sum(vent_low_mask) + np.sum(spect_low_mask))

        return {
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p_value,
            'dice_high_ventilation': dice_high,
            'dice_low_ventilation': dice_low,
            'metrics_details': {
                'vent_median': vent_median,
                'spect_median': spect_median
            }
        }