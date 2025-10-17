"""
Input/Output Utilities

This module provides utilities for loading and saving various data formats
used in the Vent4D-Mech framework, including medical images, configuration
files, and results data.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging
from pathlib import Path

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

import json
import pickle


class IOUtils:
    """
    Utility functions for data input/output operations.

    This class provides comprehensive utilities for loading and saving various
    data formats used in the Vent4D-Mech framework, with special support for
    medical imaging formats and scientific data.

    Attributes:
        logger (logging.Logger): Logger instance
        supported_formats (list): List of supported file formats
    """

    def __init__(self):
        """Initialize IOUtils."""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.nii', '.nii.gz', '.h5', '.hdf5', '.json', '.pkl', '.npz']

    def load_medical_image(self, file_path: Union[str, Path],
                          return_header: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Load medical image file.

        Args:
            file_path: Path to medical image file
            return_header: Whether to return image header information

        Returns:
            Image data array, optionally with header

        Raises:
            ImportError: If required libraries are not available
            FileNotFoundError: If file does not exist
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for medical image loading")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            img = nib.load(str(file_path))
            data = img.get_fdata().astype(np.float32)

            if return_header:
                header_info = {
                    'affine': img.affine,
                    'header': img.header,
                    'shape': img.shape,
                    'voxel_size': img.header.get_zooms()
                }
                return data, header_info
            else:
                return data

        except Exception as e:
            self.logger.error(f"Failed to load medical image {file_path}: {str(e)}")
            raise

    def save_medical_image(self, data: np.ndarray, file_path: Union[str, Path],
                         reference_image: Optional[Union[str, Path, Any]] = None,
                         affine: Optional[np.ndarray] = None) -> None:
        """
        Save medical image file.

        Args:
            data: Image data array
            file_path: Output file path
            reference_image: Reference image for header information
            affine: Affine transformation matrix

        Raises:
            ImportError: If required libraries are not available
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for medical image saving")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Determine affine matrix
            if reference_image is not None:
                if isinstance(reference_image, (str, Path)):
                    ref_img = nib.load(str(reference_image))
                    affine_matrix = ref_img.affine
                    header = ref_img.header
                else:
                    # Assume it's a nibabel image object
                    affine_matrix = reference_image.affine
                    header = reference_image.header
            elif affine is not None:
                affine_matrix = affine
                header = nib.Nifti1Header()
                header.set_zooms((1.0, 1.0, 1.0))
            else:
                # Use identity matrix
                affine_matrix = np.eye(4)
                header = nib.Nifti1Header()
                header.set_zooms((1.0, 1.0, 1.0))

            # Create and save image
            img = nib.Nifti1Image(data, affine_matrix, header)
            nib.save(img, str(file_path))

            self.logger.info(f"Saved medical image to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save medical image {file_path}: {str(e)}")
            raise

    def load_hdf5(self, file_path: Union[str, Path],
                  dataset_path: Optional[str] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load data from HDF5 file.

        Args:
            file_path: Path to HDF5 file
            dataset_path: Specific dataset path (if None, load all datasets)

        Returns:
            Loaded data

        Raises:
            ImportError: If h5py is not available
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for HDF5 operations")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        try:
            with h5py.File(file_path, 'r') as f:
                if dataset_path is not None:
                    if dataset_path in f:
                        return f[dataset_path][:]
                    else:
                        raise KeyError(f"Dataset {dataset_path} not found in file")
                else:
                    # Load all datasets
                    data = {}
                    def visitor(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            data[name] = obj[:]
                    f.visititems(visitor)
                    return data

        except Exception as e:
            self.logger.error(f"Failed to load HDF5 file {file_path}: {str(e)}")
            raise

    def save_hdf5(self, data: Union[np.ndarray, Dict[str, np.ndarray]],
                  file_path: Union[str, Path],
                  dataset_path: Optional[str] = None) -> None:
        """
        Save data to HDF5 file.

        Args:
            data: Data to save
            file_path: Output file path
            dataset_path: Dataset path (required if data is single array)

        Raises:
            ImportError: If h5py is not available
        """
        if not H5PY_AVAILABLE:
            raise ImportError("h5py is required for HDF5 operations")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with h5py.File(file_path, 'w') as f:
                if isinstance(data, dict):
                    # Save multiple datasets
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
                else:
                    # Save single dataset
                    if dataset_path is None:
                        raise ValueError("dataset_path is required when saving single array")
                    f.create_dataset(dataset_path, data=data)

            self.logger.info(f"Saved HDF5 file to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save HDF5 file {file_path}: {str(e)}")
            raise

    def load_json(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Loaded JSON data

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load JSON file {file_path}: {str(e)}")
            raise

    def save_json(self, data: Dict[str, Any], file_path: Union[str, Path],
                  indent: int = 2) -> None:
        """
        Save data to JSON file.

        Args:
            data: Data to save
            file_path: Output file path
            indent: JSON indentation level
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert numpy arrays to lists for JSON serialization
            json_data = self._convert_numpy_to_json(data)

            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=indent)

            self.logger.info(f"Saved JSON file to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save JSON file {file_path}: {str(e)}")
            raise

    def load_pickle(self, file_path: Union[str, Path]) -> Any:
        """
        Load pickle file.

        Args:
            file_path: Path to pickle file

        Returns:
            Loaded data

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load pickle file {file_path}: {str(e)}")
            raise

    def save_pickle(self, data: Any, file_path: Union[str, Path]) -> None:
        """
        Save data to pickle file.

        Args:
            data: Data to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)

            self.logger.info(f"Saved pickle file to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save pickle file {file_path}: {str(e)}")
            raise

    def load_numpy(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load numpy array file.

        Args:
            file_path: Path to numpy file (.npy or .npz)

        Returns:
            Loaded numpy array

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NumPy file not found: {file_path}")

        try:
            if file_path.suffix == '.npz':
                # Load .npz file
                data = np.load(file_path)
                # Return the first array (or dictionary if multiple)
                if len(data.files) == 1:
                    return data[data.files[0]]
                else:
                    return {key: data[key] for key in data.files}
            else:
                # Load .npy file
                return np.load(file_path)

        except Exception as e:
            self.logger.error(f"Failed to load numpy file {file_path}: {str(e)}")
            raise

    def save_numpy(self, data: np.ndarray, file_path: Union[str, Path]) -> None:
        """
        Save numpy array file.

        Args:
            data: Array to save
            file_path: Output file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if file_path.suffix == '.npz' and isinstance(data, dict):
                # Save multiple arrays
                np.savez_compressed(file_path, **data)
            else:
                # Save single array
                np.save(file_path, data)

            self.logger.info(f"Saved numpy file to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to save numpy file {file_path}: {str(e)}")
            raise

    def _convert_numpy_to_json(self, obj: Any) -> Any:
        """
        Convert numpy objects to JSON-serializable types.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(item) for item in obj]
        else:
            return obj

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information.

        Args:
            file_path: Path to file

        Returns:
            File information dictionary
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {'exists': False}

        stat = file_path.stat()

        return {
            'exists': True,
            'size': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': stat.st_mtime,
            'extension': file_path.suffix,
            'is_medical_image': file_path.suffix.lower() in ['.nii', '.nii.gz']
        }

    def __repr__(self) -> str:
        """String representation of the IOUtils instance."""
        return f"IOUtils(supported_formats={self.supported_formats})"