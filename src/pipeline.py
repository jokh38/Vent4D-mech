"""
Main Vent4D-Mech Pipeline

This module implements the main pipeline for the Vent4D-Mech framework,
orchestrating all components from image registration to ventilation analysis.
"""

from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import logging
from pathlib import Path

from .core.registration import ImageRegistration
from .core.deformation import DeformationAnalyzer
from .core.mechanical import MechanicalModeler
from .core.inverse import YoungsModulusEstimator
from .core.microstructure import MicrostructureDB
from .core.fem import FEMWorkflow
from .core.ventilation import VentilationCalculator
from .config import ConfigManager
from .utils import IOUtils, ValidationUtils, PerformanceUtils


class Vent4DMechPipeline:
    """
    Main pipeline for Vent4D-Mech framework.

    This class orchestrates the complete workflow from 4D-CT images to
    ventilation analysis, including image registration, deformation analysis,
    mechanical modeling, and ventilation calculation.

    Attributes:
        config (ConfigManager): Configuration manager
        io_utils (IOUtils): Input/output utilities
        validation_utils (ValidationUtils): Validation utilities
        performance_utils (PerformanceUtils): Performance monitoring
        components (dict): Pipeline components
        logger (logging.Logger): Logger instance
        results (dict): Pipeline results
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize Vent4D-Mech pipeline.

        Args:
            config_file: Optional configuration file path
        """
        # Initialize components
        self.config = ConfigManager(config_file)
        self.io_utils = IOUtils()
        self.validation_utils = ValidationUtils()
        self.performance_utils = PerformanceUtils()

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize pipeline components
        self.components = {}
        self._initialize_components()

        # Results storage
        self.results = {}

        self.logger.info("Initialized Vent4D-Mech pipeline")

    def _initialize_components(self) -> None:
        """
        Initialize all pipeline components.
        """
        try:
            # Image registration
            self.components['registration'] = ImageRegistration(
                method=self.config.get('registration.method', 'voxelmorph'),
                config=self.config.get_section('registration'),
                gpu=self.config.get('performance.gpu_acceleration', True)
            )

            # Deformation analysis
            self.components['deformation'] = DeformationAnalyzer(
                config=self.config.get_section('deformation'),
                gpu=self.config.get('performance.gpu_acceleration', True)
            )

            # Mechanical modeler
            self.components['mechanical'] = MechanicalModeler(
                config=self.config.get_section('mechanical')
            )

            # Young's modulus estimator
            self.components['inverse'] = YoungsModulusEstimator(
                config=self.config.get_section('inverse')
            )

            # Microstructure database
            self.components['microstructure'] = MicrostructureDB(
                config=self.config.get_section('microstructure')
            )

            # FEM workflow
            self.components['fem'] = FEMWorkflow(
                config=self.config.get_section('fem')
            )

            # Ventilation calculator
            self.components['ventilation'] = VentilationCalculator(
                config=self.config.get_section('ventilation')
            )

            self.logger.info("Initialized all pipeline components")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def load_data(self, inhale_path: Union[str, Path], exhale_path: Union[str, Path],
                  lung_mask_path: Optional[Union[str, Path]] = None,
                  voxel_spacing: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Load 4D-CT data for analysis.

        Args:
            inhale_path: Path to inhale phase CT
            exhale_path: Path to exhale phase CT
            lung_mask_path: Optional path to lung segmentation mask
            voxel_spacing: Optional voxel spacing (mm)
        """
        self.logger.info("Loading 4D-CT data...")

        try:
            # Load images
            inhale_image, inhale_header = self.io_utils.load_medical_image(
                inhale_path, return_header=True
            )
            exhale_image, exhale_header = self.io_utils.load_medical_image(
                exhale_path, return_header=True
            )

            # Validate image dimensions
            if inhale_image.shape != exhale_image.shape:
                raise ValueError("Inhale and exhale images must have same dimensions")

            # Load lung mask if provided
            lung_mask = None
            if lung_mask_path:
                lung_mask = self.io_utils.load_medical_image(lung_mask_path)
                if lung_mask.shape != inhale_image.shape:
                    raise ValueError("Lung mask must have same dimensions as CT images")

            # Determine voxel spacing
            if voxel_spacing is None:
                voxel_spacing = inhale_header.get('voxel_size', (1.0, 1.0, 1.0))[:3]

            # Store data
            self.data = {
                'inhale_image': inhale_image,
                'exhale_image': exhale_image,
                'lung_mask': lung_mask,
                'voxel_spacing': voxel_spacing,
                'image_headers': {
                    'inhale': inhale_header,
                    'exhale': exhale_header
                }
            }

            self.logger.info(f"Loaded 4D-CT data with shape {inhale_image.shape} and spacing {voxel_spacing}")

        except Exception as e:
            self.logger.error(f"Failed to load data: {str(e)}")
            raise

    def run_pipeline(self, stages: Optional[list] = None) -> Dict[str, Any]:
        """
        Run the complete Vent4D-Mech pipeline.

        Args:
            stages: List of stages to run (if None, run all stages)

        Returns:
            Pipeline results

        Raises:
            RuntimeError: If data is not loaded or pipeline fails
        """
        if not hasattr(self, 'data'):
            raise RuntimeError("Data must be loaded before running pipeline")

        # Define all available stages
        all_stages = [
            'registration',
            'deformation_analysis',
            'material_estimation',
            'fem_simulation',
            'ventilation_analysis'
        ]

        # Filter stages if specified
        if stages is None:
            stages = all_stages
        else:
            # Validate stage names
            invalid_stages = set(stages) - set(all_stages)
            if invalid_stages:
                raise ValueError(f"Invalid stages: {invalid_stages}")

        self.logger.info(f"Running pipeline stages: {stages}")

        # Start performance monitoring
        self.performance_utils.start_monitoring()

        try:
            # Stage 1: Image registration
            if 'registration' in stages:
                self.logger.info("Stage 1: Image registration")
                registration_results = self.components['registration'].register_images(
                    fixed_image=self.data['inhale_image'],
                    moving_image=self.data['exhale_image'],
                    mask=self.data['lung_mask']
                )
                self.results['registration'] = registration_results

            # Stage 2: Deformation analysis
            if 'deformation_analysis' in stages:
                self.logger.info("Stage 2: Deformation analysis")
                deformation_results = self.components['deformation'].analyze_deformation(
                    dvf=self.results['registration']['dvf'],
                    voxel_spacing=self.data['voxel_spacing'],
                    mask=self.data['lung_mask']
                )
                self.results['deformation'] = deformation_results

            # Stage 3: Material property estimation
            if 'material_estimation' in stages:
                self.logger.info("Stage 3: Material property estimation")

                # Get material constraints from microstructure database
                if self.data['lung_mask'] is not None:
                    # Estimate material properties from CT intensities
                    hu_values = self.data['inhale_image'] * self.data['lung_mask']
                    material_properties = self.components['microstructure'].estimate_properties(
                        hu_values=hu_values,
                        method='surrogate'
                    )
                else:
                    # Use default properties
                    material_properties = {
                        'youngs_modulus': np.ones(self.data['inhale_image'].shape) * 5.0,
                        'poisson_ratio': 0.45
                    }

                # Estimate Young's modulus using inverse problem
                modulus_results = self.components['inverse'].estimate_modulus(
                    observed_strain=self.results['deformation']['strain_tensor'],
                    deformation_gradient=self.results['deformation']['deformation_gradient'],
                    mask=self.data['lung_mask']
                )
                self.results['material_estimation'] = {
                    'microstructure_properties': material_properties,
                    'inverse_estimation': modulus_results
                }

            # Stage 4: FEM simulation
            if 'fem_simulation' in stages and self.data['lung_mask'] is not None:
                self.logger.info("Stage 4: FEM simulation")
                fem_results = self.components['fem'].run_simulation(
                    lung_mask=self.data['lung_mask'],
                    displacement_field=self.results['registration']['dvf'],
                    material_properties=self.results['material_estimation']['inverse_estimation']['youngs_modulus'],
                    voxel_spacing=self.data['voxel_spacing']
                )
                self.results['fem_simulation'] = fem_results

            # Stage 5: Ventilation analysis
            if 'ventilation_analysis' in stages:
                self.logger.info("Stage 5: Ventilation analysis")
                ventilation_results = self.components['ventilation'].compute_ventilation(
                    deformation_gradient=self.results['deformation']['deformation_gradient'],
                    lung_mask=self.data['lung_mask']
                )
                self.results['ventilation'] = ventilation_results

            # Stop performance monitoring
            performance_stats = self.performance_utils.stop_monitoring()
            self.results['performance'] = performance_stats

            self.logger.info("Pipeline completed successfully")
            return self.results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

    def save_results(self, output_dir: Union[str, Path],
                    save_intermediate: bool = True) -> None:
        """
        Save pipeline results.

        Args:
            output_dir: Output directory
            save_intermediate: Whether to save intermediate results
        """
        if not self.results:
            raise RuntimeError("No results to save. Run pipeline first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Saving results to {output_dir}")

        try:
            # Save main results
            if 'ventilation' in self.results:
                self.components['ventilation'].save_results(output_dir / 'ventilation')

            if 'material_estimation' in self.results:
                self.io_utils.save_json(
                    self.results['material_estimation'],
                    output_dir / 'material_estimation.json'
                )

            # Save intermediate results if requested
            if save_intermediate:
                if 'registration' in self.results:
                    self.io_utils.save_numpy(
                        self.results['registration']['dvf'],
                        output_dir / 'registration' / 'dvf.npy'
                    )

                if 'deformation' in self.results:
                    deformation_dir = output_dir / 'deformation'
                    deformation_dir.mkdir(exist_ok=True)
                    self.io_utils.save_numpy(
                        self.results['deformation']['strain_tensor'],
                        deformation_dir / 'strain_tensor.npz'
                    )

            # Save configuration and metadata
            self.config.save_config(output_dir / 'pipeline_config.yaml')
            self.io_utils.save_json(
                self.results.get('performance', {}),
                output_dir / 'performance_stats.json'
            )

            self.logger.info("Results saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    def get_results(self, stage: Optional[str] = None) -> Union[Dict[str, Any], Any]:
        """
        Get pipeline results.

        Args:
            stage: Specific stage results (if None, return all results)

        Returns:
            Results for specified stage or all results
        """
        if stage is None:
            return self.results
        else:
            return self.results.get(stage)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get pipeline execution summary.

        Returns:
            Execution summary
        """
        summary = {
            'completed_stages': list(self.results.keys()),
            'config_sources': self.config.config_sources,
            'data_shape': self.data['inhale_image'].shape if hasattr(self, 'data') else None,
            'voxel_spacing': self.data['voxel_spacing'] if hasattr(self, 'data') else None
        }

        if 'performance' in self.results:
            summary['performance'] = self.results['performance']

        return summary

    def __repr__(self) -> str:
        """String representation of the Vent4DMechPipeline instance."""
        return (f"Vent4DMechPipeline(stages_completed={list(self.results.keys())}, "
                f"has_data={hasattr(self, 'data')})")