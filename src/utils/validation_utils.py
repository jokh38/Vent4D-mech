"""
Validation Utilities

This module provides data validation and quality checking utilities for the Vent4D-Mech framework,
including medical image validation, tensor validation, and data integrity checks.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from .logging_utils import LoggingUtils


class ValidationUtils:
    """
    Data validation and quality checking utilities.

    This class provides comprehensive validation capabilities for medical images,
    numerical arrays, and configuration parameters used in the Vent4D-Mech framework.

    Attributes:
        logger (LoggingUtils): Logger instance
        validation_results (list): History of validation results
    """

    def __init__(self, logger: Optional[LoggingUtils] = None):
        """
        Initialize ValidationUtils.

        Args:
            logger: Optional logger instance (creates default if None)
        """
        self.logger = logger or LoggingUtils('validation_utils')
        self.validation_results = []

    def validate_medical_image(self, image_data: np.ndarray,
                              voxel_size: Optional[Tuple[float, ...]] = None,
                              modality: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate medical image data.

        Args:
            image_data: Medical image array
            voxel_size: Voxel spacing (dx, dy, dz)
            modality: Image modality (CT, MRI, etc.)

        Returns:
            Validation result dictionary

        Example:
            result = validator.validate_medical_image(
                ct_image, voxel_size=(1.0, 1.0, 2.5), modality="CT"
            )
        """
        validation_result = {
            'validation_type': 'medical_image',
            'passed': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }

        try:
            # Basic array validation
            if not isinstance(image_data, np.ndarray):
                validation_result['errors'].append("Image data must be a numpy array")
                validation_result['passed'] = False
                return validation_result

            # Dimension validation
            if image_data.ndim < 2 or image_data.ndim > 4:
                validation_result['errors'].append(
                    f"Image must have 2-4 dimensions, got {image_data.ndim}"
                )
                validation_result['passed'] = False

            # Size validation
            if image_data.size == 0:
                validation_result['errors'].append("Image cannot be empty")
                validation_result['passed'] = False

            # Store image information
            validation_result['info'] = {
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'size_mb': image_data.nbytes / 1024 / 1024,
                'min_value': float(np.min(image_data)) if image_data.size > 0 else None,
                'max_value': float(np.max(image_data)) if image_data.size > 0 else None,
                'mean_value': float(np.mean(image_data)) if image_data.size > 0 else None
            }

            # Check for problematic values
            if np.any(np.isnan(image_data)):
                validation_result['warnings'].append("Image contains NaN values")

            if np.any(np.isinf(image_data)):
                validation_result['warnings'].append("Image contains infinite values")

            # Modality-specific validation
            if modality:
                self._validate_modality_specific(image_data, modality, validation_result)

            # Voxel size validation
            if voxel_size:
                self._validate_voxel_size(voxel_size, validation_result)

            # Memory efficiency check
            if image_data.nbytes > 1024 * 1024 * 1024:  # > 1GB
                validation_result['warnings'].append(
                    f"Large image size: {image_data.nbytes / 1024 / 1024 / 1024:.2f} GB"
                )

        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            validation_result['passed'] = False

        self.validation_results.append(validation_result)
        return validation_result

    def _validate_modality_specific(self, image_data: np.ndarray,
                                  modality: str, validation_result: Dict[str, Any]) -> None:
        """Validate modality-specific image characteristics."""
        modality = modality.upper()

        if modality == 'CT':
            # Check Hounsfield units range
            min_val, max_val = validation_result['info']['min_value'], validation_result['info']['max_value']
            if min_val < -2000 or max_val > 4000:
                validation_result['warnings'].append(
                    f"CT values outside typical Hounsfield range (-2000 to 4000): "
                    f"min={min_val:.1f}, max={max_val:.1f}"
                )

        elif modality == 'MRI':
            # Check for reasonable MRI signal ranges
            if image_data.dtype in [np.int16, np.int32]:
                validation_result['warnings'].append(
                    "MRI image has integer dtype - consider float for quantitative analysis"
                )

    def _validate_voxel_size(self, voxel_size: Tuple[float, ...],
                            validation_result: Dict[str, Any]) -> None:
        """Validate voxel size parameters."""
        if not all(v > 0 for v in voxel_size):
            validation_result['errors'].append("All voxel sizes must be positive")
            validation_result['passed'] = False

        if len(voxel_size) > 3:
            validation_result['warnings'].append(
                f"Unusual voxel size dimensions: {len(voxel_size)} (expected 2-3)"
            )

        validation_result['info']['voxel_size'] = voxel_size

    def validate_tensor(self, tensor: np.ndarray, tensor_type: str = "general",
                        expected_shape: Optional[Tuple[int, ...]] = None) -> Dict[str, Any]:
        """
        Validate tensor data for mechanical computations.

        Args:
            tensor: Input tensor array
            tensor_type: Type of tensor (strain, stress, deformation_gradient, etc.)
            expected_shape: Expected tensor shape (optional)

        Returns:
            Validation result dictionary

        Example:
            result = validator.validate_tensor(
                strain_tensor, tensor_type="strain", expected_shape=(64, 64, 64, 3, 3)
            )
        """
        validation_result = {
            'validation_type': f'tensor_{tensor_type}',
            'passed': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }

        try:
            # Basic array validation
            if not isinstance(tensor, np.ndarray):
                validation_result['errors'].append("Tensor must be a numpy array")
                validation_result['passed'] = False
                return validation_result

            # Store tensor information
            validation_result['info'] = {
                'shape': tensor.shape,
                'dtype': str(tensor.dtype),
                'size_mb': tensor.nbytes / 1024 / 1024,
                'min_value': float(np.min(tensor)) if tensor.size > 0 else None,
                'max_value': float(np.max(tensor)) if tensor.size > 0 else None,
                'norm': float(np.linalg.norm(tensor)) if tensor.size > 0 else None
            }

            # Tensor type specific validation
            if tensor_type == "strain":
                self._validate_strain_tensor(tensor, validation_result)
            elif tensor_type == "stress":
                self._validate_stress_tensor(tensor, validation_result)
            elif tensor_type == "deformation_gradient":
                self._validate_deformation_gradient(tensor, validation_result)
            elif tensor_type == "displacement_field":
                self._validate_displacement_field(tensor, validation_result)

            # Expected shape validation
            if expected_shape and tensor.shape != expected_shape:
                validation_result['errors'].append(
                    f"Expected shape {expected_shape}, got {tensor.shape}"
                )
                validation_result['passed'] = False

            # Check for problematic values
            if np.any(np.isnan(tensor)):
                validation_result['errors'].append("Tensor contains NaN values")
                validation_result['passed'] = False

            if np.any(np.isinf(tensor)):
                validation_result['errors'].append("Tensor contains infinite values")
                validation_result['passed'] = False

        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            validation_result['passed'] = False

        self.validation_results.append(validation_result)
        return validation_result

    def _validate_strain_tensor(self, tensor: np.ndarray, validation_result: Dict[str, Any]) -> None:
        """Validate strain tensor characteristics."""
        if tensor.ndim < 3 or tensor.shape[-2:] != (3, 3):
            validation_result['errors'].append(
                f"Strain tensor must have shape (..., 3, 3), got {tensor.shape}"
            )
            validation_result['passed'] = False

        # Check for reasonable strain values (typically -0.5 to 2.0)
        max_strain = np.max(np.abs(tensor))
        if max_strain > 5.0:
            validation_result['warnings'].append(
                f"Large strain values detected: max absolute strain = {max_strain:.2f}"
            )

    def _validate_stress_tensor(self, tensor: np.ndarray, validation_result: Dict[str, Any]) -> None:
        """Validate stress tensor characteristics."""
        if tensor.ndim < 3 or tensor.shape[-2:] != (3, 3):
            validation_result['errors'].append(
                f"Stress tensor must have shape (..., 3, 3), got {tensor.shape}"
            )
            validation_result['passed'] = False

        # Check for reasonable stress values (typically -1000 to 1000 kPa for lung tissue)
        max_stress = np.max(np.abs(tensor))
        if max_stress > 10000:  # 10 MPa
            validation_result['warnings'].append(
                f"Very large stress values detected: max absolute stress = {max_stress:.2f} kPa"
            )

    def _validate_deformation_gradient(self, tensor: np.ndarray, validation_result: Dict[str, Any]) -> None:
        """Validate deformation gradient tensor characteristics."""
        if tensor.ndim < 3 or tensor.shape[-2:] != (3, 3):
            validation_result['errors'].append(
                f"Deformation gradient must have shape (..., 3, 3), got {tensor.shape}"
            )
            validation_result['passed'] = False

        # Check determinant (Jacobian) values
        if tensor.ndim == 5:  # 3D volume of 3x3 matrices
            jacobians = np.linalg.det(tensor)
            if np.any(jacobians <= 0):
                validation_result['warnings'].append(
                    "Non-positive Jacobian determinants detected (potential material inversion)"
                )

            if np.any(jacobians > 10):
                validation_result['warnings'].append(
                    "Large Jacobian determinants detected (potential extreme deformation)"
                )

    def _validate_displacement_field(self, tensor: np.ndarray, validation_result: Dict[str, Any]) -> None:
        """Validate displacement field characteristics."""
        if tensor.ndim < 4 or tensor.shape[-1] != 3:
            validation_result['errors'].append(
                f"Displacement field must have shape (..., 3), got {tensor.shape}"
            )
            validation_result['passed'] = False

        # Check for reasonable displacement magnitudes
        if tensor.ndim == 4:  # 3D volume with 3 displacement components
            displacement_magnitudes = np.linalg.norm(tensor, axis=-1)
            max_displacement = np.max(displacement_magnitudes)

            # For lung CT, displacements are typically < 100mm
            if max_displacement > 200:
                validation_result['warnings'].append(
                    f"Large displacement magnitudes detected: max = {max_displacement:.2f} mm"
                )

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Validation result dictionary
        """
        validation_result = {
            'validation_type': 'config',
            'passed': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }

        try:
            if not isinstance(config, dict):
                validation_result['errors'].append("Configuration must be a dictionary")
                validation_result['passed'] = False
                return validation_result

            # Check required sections
            required_sections = ['registration', 'mechanical', 'performance']
            for section in required_sections:
                if section not in config:
                    validation_result['warnings'].append(f"Missing config section: {section}")

            # Validate mechanical configuration
            if 'mechanical' in config:
                self._validate_mechanical_config(config['mechanical'], validation_result)

            # Validate registration configuration
            if 'registration' in config:
                self._validate_registration_config(config['registration'], validation_result)

            validation_result['info'] = {
                'total_sections': len(config),
                'sections': list(config.keys())
            }

        except Exception as e:
            validation_result['errors'].append(f"Configuration validation failed: {str(e)}")
            validation_result['passed'] = False

        self.validation_results.append(validation_result)
        return validation_result

    def _validate_mechanical_config(self, mechanical_config: Dict[str, Any],
                                   validation_result: Dict[str, Any]) -> None:
        """Validate mechanical configuration section."""
        if 'constitutive_model' in mechanical_config:
            model = mechanical_config['constitutive_model']
            valid_models = ['neo_hookean', 'mooney_rivlin', 'yeoh', 'ogden', 'linear_elastic']
            if model not in valid_models:
                validation_result['warnings'].append(
                    f"Unknown constitutive model: {model}. Valid options: {valid_models}"
                )

        if 'material_parameters' in mechanical_config:
            params = mechanical_config['material_parameters']
            if not isinstance(params, dict):
                validation_result['errors'].append("Material parameters must be a dictionary")
                validation_result['passed'] = False

    def _validate_registration_config(self, registration_config: Dict[str, Any],
                                    validation_result: Dict[str, Any]) -> None:
        """Validate registration configuration section."""
        if 'method' in registration_config:
            method = registration_config['method']
            valid_methods = ['voxelmorph', 'simpleitk', 'deformable']
            if method not in valid_methods:
                validation_result['warnings'].append(
                    f"Unknown registration method: {method}. Valid options: {valid_methods}"
                )

    def validate_file_path(self, file_path: Union[str, Path],
                          expected_extensions: Optional[List[str]] = None,
                          must_exist: bool = True) -> Dict[str, Any]:
        """
        Validate file path.

        Args:
            file_path: File path to validate
            expected_extensions: List of expected file extensions
            must_exist: Whether file must exist

        Returns:
            Validation result dictionary
        """
        validation_result = {
            'validation_type': 'file_path',
            'passed': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }

        try:
            file_path = Path(file_path)

            validation_result['info'] = {
                'path': str(file_path),
                'exists': file_path.exists(),
                'is_file': file_path.is_file() if file_path.exists() else None,
                'extension': file_path.suffix.lower(),
                'size_mb': file_path.stat().st_size / 1024 / 1024 if file_path.exists() else None
            }

            if must_exist and not file_path.exists():
                validation_result['errors'].append(f"File does not exist: {file_path}")
                validation_result['passed'] = False

            if expected_extensions and file_path.suffix.lower() not in expected_extensions:
                validation_result['errors'].append(
                    f"File extension {file_path.suffix} not in expected: {expected_extensions}"
                )
                validation_result['passed'] = False

        except Exception as e:
            validation_result['errors'].append(f"File path validation failed: {str(e)}")
            validation_result['passed'] = False

        self.validation_results.append(validation_result)
        return validation_result

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation results.

        Returns:
            Validation summary
        """
        if not self.validation_results:
            return {'message': 'No validation results available'}

        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r['passed'])
        failed_validations = total_validations - passed_validations

        # Count validation types
        validation_types = {}
        for result in self.validation_results:
            vtype = result['validation_type']
            validation_types[vtype] = validation_types.get(vtype, 0) + 1

        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'failed_validations': failed_validations,
            'success_rate': passed_validations / total_validations * 100,
            'validation_types': validation_types,
            'recent_results': self.validation_results[-5:]  # Last 5 results
        }

    def clear_validation_history(self) -> None:
        """Clear validation history."""
        self.validation_results.clear()
        self.logger.debug("Validation history cleared")

    def __repr__(self) -> str:
        """String representation of the ValidationUtils instance."""
        return f"ValidationUtils(validations_performed={len(self.validation_results)})"


# Convenience function for getting a validator
def get_validator(logger: Optional[LoggingUtils] = None) -> ValidationUtils:
    """
    Get a configured validator instance.

    Args:
        logger: Optional logger instance

    Returns:
        Configured ValidationUtils instance
    """
    return ValidationUtils(logger=logger)


# Module-level validator instance
default_validator = get_validator()