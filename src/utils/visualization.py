"""
Visualization Utilities

This module provides visualization utilities for the Vent4D-Mech framework,
including medical image visualization, plotting, and 3D visualization capabilities.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List
import warnings
from pathlib import Path

from .logging_utils import LoggingUtils

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False


class Visualization:
    """
    Visualization utilities for medical images and biomechanical data.

    This class provides comprehensive visualization capabilities for medical images,
    tensor fields, and analysis results with support for both 2D and 3D visualization.

    Attributes:
        logger (LoggingUtils): Logger instance
        figure_size (tuple): Default figure size for plots
        colormap (str): Default colormap for images
    """

    def __init__(self, logger: Optional[LoggingUtils] = None,
                 figure_size: Tuple[int, int] = (10, 8),
                 colormap: str = 'gray'):
        """
        Initialize Visualization.

        Args:
            logger: Optional logger instance
            figure_size: Default figure size (width, height)
            colormap: Default colormap for image display
        """
        self.logger = logger or LoggingUtils('visualization')
        self.figure_size = figure_size
        self.colormap = colormap

        # Set matplotlib backend if available
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            self.logger.debug("Matplotlib backend configured")

    def display_image_slice(self, image: np.ndarray, slice_index: Optional[int] = None,
                          axis: int = 0, title: Optional[str] = None,
                          colormap: Optional[str] = None, show_colorbar: bool = True,
                          save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Display a single slice of a 3D image.

        Args:
            image: 3D image array
            slice_index: Slice index (middle slice if None)
            axis: Axis along which to take slice (0=sagittal, 1=coronal, 2=axial)
            title: Optional plot title
            colormap: Optional colormap override
            show_colorbar: Whether to show colorbar
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure if successful, None otherwise

        Example:
            fig = viz.display_image_slice(
                ct_image, slice_index=100, axis=2, title="Axial CT Slice"
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for visualization")
            return None

        if image.ndim < 3:
            self.logger.error("Image must be at least 3D for slice display")
            return None

        # Determine slice index
        if slice_index is None:
            slice_index = image.shape[axis] // 2

        if slice_index < 0 or slice_index >= image.shape[axis]:
            self.logger.error(f"Slice index {slice_index} out of range for axis {axis}")
            return None

        # Extract slice
        slice_data = np.take(image, slice_index, axis=axis)

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Display slice
        im = ax.imshow(slice_data, cmap=colormap or self.colormap, aspect='auto')

        if show_colorbar:
            plt.colorbar(im, ax=ax, label='Intensity')

        # Set title
        if title:
            ax.set_title(title)
        else:
            axis_names = ['Sagittal', 'Coronal', 'Axial']
            ax.set_title(f'{axis_names[axis]} Slice {slice_index}')

        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")

        self.logger.debug(f"Displayed slice {slice_index} along axis {axis}")
        return fig

    def display_image_montage(self, image: np.ndarray, slices: Optional[List[int]] = None,
                            axis: int = 0, rows: int = 4, cols: int = 5,
                            title: Optional[str] = None, colormap: Optional[str] = None,
                            save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Display multiple slices in a montage.

        Args:
            image: 3D image array
            slices: List of slice indices (evenly spaced if None)
            axis: Axis along which to take slices
            rows: Number of rows in montage
            cols: Number of columns in montage
            title: Optional plot title
            colormap: Optional colormap override
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure if successful, None otherwise

        Example:
            fig = viz.display_image_montage(
                ct_image, axis=2, rows=3, cols=4, title="CT Montage"
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for visualization")
            return None

        if image.ndim < 3:
            self.logger.error("Image must be at least 3D for montage display")
            return None

        # Determine slice indices
        if slices is None:
            total_slices = rows * cols
            slice_indices = np.linspace(0, image.shape[axis] - 1, total_slices, dtype=int)
        else:
            slice_indices = slices[:rows * cols]

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, slice_idx in enumerate(slice_indices):
            if i >= len(axes):
                break

            # Extract slice
            slice_data = np.take(image, slice_idx, axis=axis)

            # Display slice
            axes[i].imshow(slice_data, cmap=colormap or self.colormap, aspect='auto')
            axes[i].set_title(f'Slice {slice_idx}')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis('off')

        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            axis_names = ['Sagittal', 'Coronal', 'Axial']
            fig.suptitle(f'{axis_names[axis]} Montage', fontsize=16)

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Montage saved to {save_path}")

        self.logger.debug(f"Created montage with {len(slice_indices)} slices")
        return fig

    def plot_tensor_field(self, tensor: np.ndarray, slice_index: Optional[int] = None,
                         axis: int = 0, component: str = 'magnitude',
                         title: Optional[str] = None, colormap: Optional[str] = None,
                         save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Visualize tensor field components.

        Args:
            tensor: Tensor field array (e.g., strain, stress)
            slice_index: Slice index (middle slice if None)
            axis: Axis along which to take slice
            component: Component to visualize ('magnitude', 'von_mises', or specific index)
            title: Optional plot title
            colormap: Optional colormap override
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure if successful, None otherwise

        Example:
            fig = viz.plot_tensor_field(
                strain_tensor, component="magnitude", title="Strain Magnitude"
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for visualization")
            return None

        if tensor.ndim < 4 or tensor.shape[-2:] != (3, 3):
            self.logger.error("Tensor must have shape (..., 3, 3)")
            return None

        # Determine slice index
        if slice_index is None:
            slice_index = tensor.shape[axis] // 2

        # Extract tensor slice
        tensor_slice = np.take(tensor, slice_index, axis=axis)

        # Compute component
        if component == 'magnitude':
            data = np.linalg.norm(tensor_slice, axis=(-2, -1))
        elif component == 'von_mises':
            # Compute von Mises equivalent stress/strain
            s11, s22, s33 = tensor_slice[..., 0, 0], tensor_slice[..., 1, 1], tensor_slice[..., 2, 2]
            s12, s13, s23 = tensor_slice[..., 0, 1], tensor_slice[..., 0, 2], tensor_slice[..., 1, 2]
            von_mises = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2 +
                                      6 * (s12**2 + s13**2 + s23**2)))
            data = von_mises
        elif component.startswith('component_'):
            # Extract specific component (e.g., component_00, component_01, etc.)
            comp_idx = component.split('_')[1]
            if len(comp_idx) == 2:
                i, j = int(comp_idx[0]), int(comp_idx[1])
                data = tensor_slice[..., i, j]
            else:
                self.logger.error(f"Invalid component specification: {component}")
                return None
        else:
            self.logger.error(f"Unknown component: {component}")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Display data
        im = ax.imshow(data, cmap=colormap or 'RdBu_r', aspect='auto')
        plt.colorbar(im, ax=ax, label=component.replace('_', ' ').title())

        # Set title
        if title:
            ax.set_title(title)
        else:
            axis_names = ['Sagittal', 'Coronal', 'Axial']
            ax.set_title(f'{axis_names[axis]} Slice {slice_index} - {component.title()}')

        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Tensor field plot saved to {save_path}")

        self.logger.debug(f"Plotted tensor field component: {component}")
        return fig

    def plot_displacement_field(self, displacement: np.ndarray, slice_index: Optional[int] = None,
                              axis: int = 0, subsample: int = 10,
                              title: Optional[str] = None, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Visualize displacement field with quiver plot.

        Args:
            displacement: Displacement field array (..., 3)
            slice_index: Slice index (middle slice if None)
            axis: Axis along which to take slice
            subsample: Subsampling factor for quiver plot
            title: Optional plot title
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure if successful, None otherwise

        Example:
            fig = viz.plot_displacement_field(
                displacement_field, subsample=5, title="Displacement Field"
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for visualization")
            return None

        if displacement.ndim < 4 or displacement.shape[-1] != 3:
            self.logger.error("Displacement field must have shape (..., 3)")
            return None

        # Determine slice index
        if slice_index is None:
            slice_index = displacement.shape[axis] // 2

        # Extract displacement slice
        disp_slice = np.take(displacement, slice_index, axis=axis)

        # Create coordinate grids
        ny, nx = disp_slice.shape[:2]
        y, x = np.mgrid[0:ny, 0:nx]

        # Subsample for quiver plot
        y_sub = y[::subsample, ::subsample]
        x_sub = x[::subsample, ::subsample]
        u_sub = disp_slice[::subsample, ::subsample, 1]  # y-component
        v_sub = disp_slice[::subsample, ::subsample, 0]  # x-component

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Calculate magnitude for background
        magnitude = np.linalg.norm(disp_slice, axis=-1)
        im = ax.imshow(magnitude, cmap='viridis', aspect='auto', alpha=0.7)

        # Add quiver plot
        quiver = ax.quiver(x_sub, y_sub, u_sub, v_sub,
                          color='red', alpha=0.8, scale=None)

        plt.colorbar(im, ax=ax, label='Displacement Magnitude')
        plt.colorbar(quiver, ax=ax, label='Displacement Components')

        # Set title
        if title:
            ax.set_title(title)
        else:
            axis_names = ['Sagittal', 'Coronal', 'Axial']
            ax.set_title(f'{axis_names[axis]} Slice {slice_index} - Displacement Field')

        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Displacement field plot saved to {save_path}")

        self.logger.debug(f"Plotted displacement field with subsample factor {subsample}")
        return fig

    def create_3d_volume_rendering(self, volume: np.ndarray, threshold: Optional[float] = None,
                                 title: Optional[str] = None, save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Create 3D volume rendering using Plotly.

        Args:
            volume: 3D volume array
            threshold: Intensity threshold for surface extraction
            title: Optional plot title
            save_path: Optional path to save figure

        Returns:
            Plotly figure if successful, None otherwise

        Example:
            fig = viz.create_3d_volume_rendering(
                ct_volume, threshold=-500, title="3D Lung Volume"
            )
        """
        if not PLOTLY_AVAILABLE:
            self.logger.error("Plotly not available for 3D visualization")
            return None

        if volume.ndim != 3:
            self.logger.error("Volume must be 3D for volume rendering")
            return None

        # Auto-detect threshold if not provided
        if threshold is None:
            threshold = np.percentile(volume[volume != 0], 50)
            self.logger.debug(f"Auto-detected threshold: {threshold:.2f}")

        self.logger.info(f"Creating 3D volume rendering with threshold {threshold}")

        # Create coordinate grids
        z, y, x = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]

        # Create figure
        fig = go.Figure(data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=volume.flatten(),
            isomin=threshold,
            isomax=volume.max(),
            opacity=0.1,
            surface_count=10,
            colorscale='Viridis'
        ))

        # Update layout
        fig.update_layout(
            title=title or '3D Volume Rendering',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=800,
            height=600
        )

        # Save figure if requested
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"3D volume rendering saved to {save_path}")

        self.logger.debug("3D volume rendering completed")
        return fig

    def plot_comparison(self, images: List[np.ndarray], titles: List[str],
                       slice_index: Optional[int] = None, axis: int = 0,
                       colormap: Optional[str] = None, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot multiple images side by side for comparison.

        Args:
            images: List of images to compare
            titles: List of titles for each image
            slice_index: Slice index (middle slice if None)
            axis: Axis along which to take slices
            colormap: Optional colormap override
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure if successful, None otherwise

        Example:
            fig = viz.plot_comparison(
                [original, processed], ["Original", "Processed"],
                slice_index=100
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for visualization")
            return None

        if len(images) != len(titles):
            self.logger.error("Number of images must match number of titles")
            return None

        # Determine slice index
        if slice_index is None:
            slice_index = images[0].shape[axis] // 2

        # Create figure
        fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 4, 4))
        if len(images) == 1:
            axes = [axes]

        for i, (image, title) in enumerate(zip(images, titles)):
            # Extract slice
            slice_data = np.take(image, slice_index, axis=axis)

            # Display slice
            im = axes[i].imshow(slice_data, cmap=colormap or self.colormap, aspect='auto')
            axes[i].set_title(title)
            axes[i].axis('off')

            # Add colorbar to first subplot
            if i == 0:
                plt.colorbar(im, ax=axes[i], label='Intensity', fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")

        self.logger.debug(f"Created comparison plot with {len(images)} images")
        return fig

    def plot_histogram(self, data: np.ndarray, bins: int = 50,
                      title: Optional[str] = None, xlabel: str = 'Value',
                      ylabel: str = 'Frequency', save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Plot histogram of data values.

        Args:
            data: Data array
            bins: Number of histogram bins
            title: Optional plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save figure

        Returns:
            Matplotlib figure if successful, None otherwise

        Example:
            fig = viz.plot_histogram(
                ct_image.flatten(), bins=100, title="CT Intensity Distribution"
            )
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.error("Matplotlib not available for visualization")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot histogram
        ax.hist(data.flatten(), bins=bins, alpha=0.7, edgecolor='black')

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or 'Data Distribution')

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure if requested
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Histogram plot saved to {save_path}")

        self.logger.debug(f"Created histogram with {bins} bins")
        return fig

    def create_visualization_report(self, data_dict: Dict[str, Any],
                                  output_dir: str, report_name: str = "visualization_report") -> str:
        """
        Create comprehensive visualization report.

        Args:
            data_dict: Dictionary containing data to visualize
            output_dir: Output directory for report
            report_name: Name of the report

        Returns:
            Path to created report

        Example:
            report_path = viz.create_visualization_report(
                {"ct_image": ct_data, "strain": strain_data},
                output_dir="./reports"
            )
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Creating visualization report: {report_name}")

        # Create subdirectory for this report
        report_dir = output_path / report_name
        report_dir.mkdir(exist_ok=True)

        figures_created = []

        # Generate visualizations for different data types
        for key, data in data_dict.items():
            if isinstance(data, np.ndarray):
                if data.ndim >= 3:
                    # 3D volume - create montage
                    fig = self.display_image_montage(
                        data, title=f"{key} Montage", save_path=str(report_dir / f"{key}_montage.png")
                    )
                    if fig:
                        plt.close(fig)
                        figures_created.append(f"{key}_montage.png")

                    # Create histogram
                    fig = self.plot_histogram(
                        data.flatten(), title=f"{key} Histogram",
                        save_path=str(report_dir / f"{key}_histogram.png")
                    )
                    if fig:
                        plt.close(fig)
                        figures_created.append(f"{key}_histogram.png")

                elif data.ndim == 2:
                    # 2D image - create single view
                    fig = self.display_image_slice(
                        data, title=f"{key}", save_path=str(report_dir / f"{key}.png")
                    )
                    if fig:
                        plt.close(fig)
                        figures_created.append(f"{key}.png")

        # Create HTML report
        html_content = self._generate_html_report(report_name, figures_created, data_dict)
        html_file = report_dir / f"{report_name}.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        self.logger.info(f"Visualization report created: {html_file}")
        return str(html_file)

    def _generate_html_report(self, report_name: str, figures: List[str],
                            data_dict: Dict[str, Any]) -> str:
        """Generate HTML content for visualization report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                .figure img {{ max-width: 800px; border: 1px solid #ddd; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{report_name}</h1>

            <div class="summary">
                <h2>Summary</h2>
                <p>Generated on: {np.datetime64('now')}</p>
                <p>Total figures: {len(figures)}</p>
                <p>Data keys: {list(data_dict.keys())}</p>
            </div>

            <h2>Visualizations</h2>
        """

        for figure in figures:
            html += f"""
            <div class="figure">
                <h3>{figure.replace('_', ' ').title()}</h3>
                <img src="{figure}" alt="{figure}">
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html

    def __repr__(self) -> str:
        """String representation of the Visualization instance."""
        return f"Visualization(figure_size={self.figure_size}, colormap='{self.colormap}')"


# Convenience function for getting visualization utilities
def get_visualization(logger: Optional[LoggingUtils] = None,
                     figure_size: Tuple[int, int] = (10, 8),
                     colormap: str = 'gray') -> Visualization:
    """
    Get a configured visualization instance.

    Args:
        logger: Optional logger instance
        figure_size: Default figure size
        colormap: Default colormap

    Returns:
        Configured Visualization instance
    """
    return Visualization(logger=logger, figure_size=figure_size, colormap=colormap)


# Module-level visualization instance
default_visualization = get_visualization()