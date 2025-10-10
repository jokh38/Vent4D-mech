"""
VoxelMorph-based Deep Learning Image Registration

This module implements VoxelMorph deep learning-based deformable image registration
using PyTorch, providing fast inference and unsupervised learning capabilities
for 4D-CT lung image registration.
"""

from typing import Optional, Dict, Any, Tuple, Union, List
import numpy as np
import time
import logging
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import voxelmorph as vxm
    VOXMORPH_AVAILABLE = True
except ImportError:
    VOXMORPH_AVAILABLE = False

from ..registration_utils import RegistrationUtils


class VoxelMorphRegistration:
    """
    VoxelMorph-based deep learning image registration.

    This class implements VoxelMorph deep learning-based registration using PyTorch,
    providing fast inference and unsupervised learning capabilities for
    4D-CT lung image registration.

    Attributes:
        config (dict): Configuration parameters
        gpu (bool): Whether to use GPU acceleration
        device (torch.device): PyTorch device
        model (nn.Module): VoxelMorph model
        utils (RegistrationUtils): Utility functions
        logger (logging.Logger): Logger instance
        is_trained (bool): Whether the model is trained
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, gpu: bool = True):
        """
        Initialize VoxelMorphRegistration instance.

        Args:
            config: Configuration parameters
            gpu: Whether to use GPU acceleration

        Raises:
            ImportError: If PyTorch or VoxelMorph is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Install with: pip install torch")
        if not VOXMORPH_AVAILABLE:
            raise ImportError("VoxelMorph is not available. Install with: pip install voxelmorph")

        self.config = config or self._get_default_config()
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.gpu else 'cpu')

        # Initialize utilities
        self.utils = RegistrationUtils()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.model = None
        self.optimizer = None
        self.is_trained = False
        self._initialize_model()

        self.logger.info(f"Initialized VoxelMorphRegistration on device: {self.device}")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration parameters.

        Returns:
            Default configuration dictionary
        """
        return {
            'model_type': 'vxm',  # 'vxm' or 'custom'
            'model_path': None,  # Path to pre-trained model
            'input_shape': [160, 192, 224],  # Input image shape
            'batch_size': 1,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'loss_weights': {
                'similarity': 1.0,
                'regularization': 0.01,
                'bending': 0.001
            },
            'architecture': {
                'encoder': [32, 64, 128, 256, 256, 256],  # Feature channels
                'decoder': [256, 256, 256, 128, 64, 32],
                'int_steps': 7,
                'int_downsize': 2
            },
            'training': {
                'epochs': 1000,
                'validation_split': 0.2,
                'early_stopping_patience': 50,
                'reduce_lr_patience': 20,
                'save_best_only': True,
                'checkpoints_dir': 'checkpoints'
            },
            'inference': {
                'resize_input': True,
                'pad_input': True,
                'interpolate_dvf': True
            }
        }

    def _initialize_model(self) -> None:
        """
        Initialize the VoxelMorph model.
        """
        # Create VoxelMorph model
        self.model = vxm.networks.VxmDense(
            inshape=self.config['input_shape'],
            nb_unet_features=[self.config['architecture']['encoder'],
                            self.config['architecture']['decoder']],
            int_steps=self.config['architecture']['int_steps'],
            int_downsize=self.config['architecture']['int_downsize']
        )

        # Move to device
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Load pre-trained model if provided
        if self.config['model_path']:
            self.load_model(self.config['model_path'])
            self.is_trained = True

        self.logger.info(f"Initialized VoxelMorph model with {sum(p.numel() for p in self.model.parameters())} parameters")

    def register_images(self, fixed_data: np.ndarray, moving_data: np.ndarray,
                       mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform image registration using trained VoxelMorph model.

        Args:
            fixed_data: Fixed image (target) as numpy array
            moving_data: Moving image (source) as numpy array
            mask: Optional mask for registration region

        Returns:
            Dictionary containing registration results

        Raises:
            RuntimeError: If model is not trained
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before inference. Call train() or load a pre-trained model.")

        start_time = time.time()
        self.logger.info("Starting VoxelMorph registration...")

        # Preprocess images
        fixed_processed, moving_processed = self._preprocess_images(fixed_data, moving_data)

        # Convert to tensors
        fixed_tensor = torch.from_numpy(fixed_processed).float().to(self.device)
        moving_tensor = torch.from_numpy(moving_processed).float().to(self.device)

        # Add batch dimension
        if fixed_tensor.dim() == 3:
            fixed_tensor = fixed_tensor.unsqueeze(0).unsqueeze(0)
        if moving_tensor.dim() == 3:
            moving_tensor = moving_tensor.unsqueeze(0).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            self.model.eval()
            dvf_tensor = self.model(fixed_tensor, moving_tensor)

        # Convert back to numpy
        dvf_array = dvf_tensor.cpu().numpy()

        # Reshape to (D, H, W, 3) format
        if dvf_array.shape[0] == 1:
            dvf_array = dvf_array[0]
        dvf_array = np.transpose(dvf_array, (1, 2, 3, 0))

        # Postprocess DVF
        dvf_array = self._postprocess_dvf(dvf_array, fixed_data.shape)

        # Apply transform to moving image
        transformed_image = self._apply_transform(moving_data, dvf_array)

        registration_time = time.time() - start_time

        results = {
            'dvf': dvf_array,
            'transformed_image': transformed_image,
            'registration_time': registration_time,
            'method': 'voxelmorph',
            'model_info': self._get_model_info()
        }

        self.logger.info(f"Registration completed in {registration_time:.2f}s")
        return results

    def _preprocess_images(self, fixed_data: np.ndarray, moving_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images for VoxelMorph inference.

        Args:
            fixed_data: Fixed image array
            moving_data: Moving image array

        Returns:
            Preprocessed fixed and moving images
        """
        fixed_processed = fixed_data.astype(np.float32)
        moving_processed = moving_data.astype(np.float32)

        # Normalize to [0, 1] range
        fixed_processed = self.utils.normalize_image(fixed_processed, method='min_max')
        moving_processed = self.utils.normalize_image(moving_processed, method='min_max')

        # Resize if needed
        if self.config['inference']['resize_input']:
            target_shape = self.config['input_shape']
            fixed_processed = self.utils.resize_image(fixed_processed, target_shape)
            moving_processed = self.utils.resize_image(moving_processed, target_shape)

        # Pad if needed
        if self.config['inference']['pad_input']:
            fixed_processed = self.utils.pad_image(fixed_processed, self.config['input_shape'])
            moving_processed = self.utils.pad_image(moving_processed, self.config['input_shape'])

        return fixed_processed, moving_processed

    def _postprocess_dvf(self, dvf: np.ndarray, original_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Postprocess displacement vector field.

        Args:
            dvf: Displacement vector field
            original_shape: Original image shape

        Returns:
            Postprocessed DVF
        """
        # Resize back to original shape if needed
        if self.config['inference']['resize_input']:
            dvf = self.utils.resize_vector_field(dvf, original_shape)

        # Interpolate if needed
        if self.config['inference']['interpolate_dvf']:
            dvf = self.utils.interpolate_vector_field(dvf, original_shape)

        return dvf

    def _apply_transform(self, moving_data: np.ndarray, dvf: np.ndarray) -> np.ndarray:
        """
        Apply displacement vector field to moving image.

        Args:
            moving_data: Moving image
            dvf: Displacement vector field

        Returns:
            Transformed image
        """
        # Use scipy for interpolation
        from scipy.ndimage import map_coordinates

        # Create coordinate grids
        z, y, x = np.mgrid[0:moving_data.shape[0], 0:moving_data.shape[1], 0:moving_data.shape[2]]

        # Add displacement
        z_new = z - dvf[..., 0]
        y_new = y - dvf[..., 1]
        x_new = x - dvf[..., 2]

        # Interpolate
        coordinates = np.array([z_new.ravel(), y_new.ravel(), x_new.ravel()])
        transformed = map_coordinates(moving_data, coordinates, order=1, mode='nearest')
        transformed = transformed.reshape(moving_data.shape)

        return transformed

    def train(self, training_data: List[Dict[str, np.ndarray]],
              validation_data: Optional[List[Dict[str, np.ndarray]]] = None,
              epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the VoxelMorph model.

        Args:
            training_data: List of training samples with 'fixed', 'moving', 'dvf' keys
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training history and metrics
        """
        epochs = epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['batch_size']

        self.logger.info(f"Starting VoxelMorph training for {epochs} epochs...")

        # Create datasets
        train_dataset = VoxelMorphDataset(training_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = None
        val_loader = None
        if validation_data:
            val_dataset = VoxelMorphDataset(validation_data)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss functions
        similarity_loss = vxm.losses.MSE().loss
        regularization_loss = vxm.losses.Grad('l2', loss_mult=2).loss

        # Training loop
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_similarity': [],
            'val_similarity': [],
            'train_regularization': [],
            'val_regularization': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch(
                train_loader, similarity_loss, regularization_loss
            )

            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate_epoch(
                    val_loader, similarity_loss, regularization_loss
                )

            # Update history
            training_history['train_loss'].append(train_metrics['total_loss'])
            training_history['train_similarity'].append(train_metrics['similarity_loss'])
            training_history['train_regularization'].append(train_metrics['regularization_loss'])

            if val_metrics:
                training_history['val_loss'].append(val_metrics['total_loss'])
                training_history['val_similarity'].append(val_metrics['similarity_loss'])
                training_history['val_regularization'].append(val_metrics['regularization_loss'])

            # Logging
            log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_metrics['total_loss']:.6f}"
            if val_metrics:
                log_msg += f", Val Loss: {val_metrics['total_loss']:.6f}"
            self.logger.info(log_msg)

            # Early stopping
            if val_metrics and val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                if self.config['training']['save_best_only']:
                    self.save_model('best_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info("Early stopping triggered")
                break

        # Mark model as trained
        self.is_trained = True

        self.logger.info("VoxelMorph training completed")
        return training_history

    def _train_epoch(self, train_loader: DataLoader, similarity_loss, regularization_loss) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            similarity_loss: Similarity loss function
            regularization_loss: Regularization loss function

        Returns:
            Training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_similarity = 0.0
        total_regularization = 0.0
        num_batches = 0

        for batch in train_loader:
            fixed, moving, target_dvf = batch
            fixed = fixed.to(self.device)
            moving = moving.to(self.device)
            target_dvf = target_dvf.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predicted_dvf = self.model(fixed, moving)

            # Compute losses
            sim_loss = similarity_loss(predicted_dvf, target_dvf)
            reg_loss = regularization_loss(predicted_dvf, target_dvf)
            total = (self.config['loss_weights']['similarity'] * sim_loss +
                    self.config['loss_weights']['regularization'] * reg_loss)

            # Backward pass
            total.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += total.item()
            total_similarity += sim_loss.item()
            total_regularization += reg_loss.item()
            num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'similarity_loss': total_similarity / num_batches,
            'regularization_loss': total_regularization / num_batches
        }

    def _validate_epoch(self, val_loader: DataLoader, similarity_loss, regularization_loss) -> Dict[str, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader
            similarity_loss: Similarity loss function
            regularization_loss: Regularization loss function

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_similarity = 0.0
        total_regularization = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                fixed, moving, target_dvf = batch
                fixed = fixed.to(self.device)
                moving = moving.to(self.device)
                target_dvf = target_dvf.to(self.device)

                # Forward pass
                predicted_dvf = self.model(fixed, moving)

                # Compute losses
                sim_loss = similarity_loss(predicted_dvf, target_dvf)
                reg_loss = regularization_loss(predicted_dvf, target_dvf)
                total = (self.config['loss_weights']['similarity'] * sim_loss +
                        self.config['loss_weights']['regularization'] * reg_loss)

                # Update metrics
                total_loss += total.item()
                total_similarity += sim_loss.item()
                total_regularization += reg_loss.item()
                num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'similarity_loss': total_similarity / num_batches,
            'regularization_loss': total_regularization / num_batches
        }

    def save_model(self, save_path: Union[str, Path]) -> None:
        """
        Save trained model to disk.

        Args:
            save_path: Path to save the model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_info': self._get_model_info()
        }, save_path)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load trained model from disk.

        Args:
            model_path: Path to the saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config.update(checkpoint.get('config', {}))

        self.is_trained = True
        self.logger.info(f"Model loaded from {model_path}")

    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Model information dictionary
        """
        return {
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'input_shape': self.config['input_shape'],
            'architecture': self.config['architecture']
        }

    def __repr__(self) -> str:
        """String representation of the VoxelMorphRegistration instance."""
        return (f"VoxelMorphRegistration(device='{self.device}', "
                f"trained={self.is_trained}, "
                f"input_shape={self.config['input_shape']})")


class VoxelMorphDataset(Dataset):
    """
    Dataset class for VoxelMorph training.
    """

    def __init__(self, data: List[Dict[str, np.ndarray]]):
        """
        Initialize dataset.

        Args:
            data: List of samples with 'fixed', 'moving', 'dvf' keys
        """
        self.data = data

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item from dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (fixed, moving, dvf) tensors
        """
        sample = self.data[idx]

        fixed = torch.from_numpy(sample['fixed']).float()
        moving = torch.from_numpy(sample['moving']).float()
        dvf = torch.from_numpy(sample['dvf']).float()

        # Add channel dimension
        if fixed.dim() == 3:
            fixed = fixed.unsqueeze(0)
        if moving.dim() == 3:
            moving = moving.unsqueeze(0)
        if dvf.dim() == 4:
            dvf = dvf.permute(3, 0, 1, 2)  # (3, D, H, W)

        return fixed, moving, dvf