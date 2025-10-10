#!/usr/bin/env python3
"""
Basic Vent4D-Mech Pipeline Example

This script demonstrates the basic usage of the Vent4D-Mech pipeline for
lung tissue dynamics modeling from 4D-CT data.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path to import vent4d_mech
sys.path.insert(0, str(Path(__file__).parent.parent))

import vent4d_mech as v4d


def setup_logging():
    """Set up logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_basic_pipeline():
    """Run a basic Vent4D-Mech pipeline example."""
    print("Vent4D-Mech Basic Pipeline Example")
    print("=" * 40)

    # Initialize the pipeline with default configuration
    print("\n1. Initializing pipeline...")
    pipeline = v4d.Vent4DMechPipeline(
        config_file="config/default.yaml"
    )

    # Load 4D-CT data
    print("\n2. Loading 4D-CT data...")
    # Note: Replace these paths with your actual data paths
    try:
        pipeline.load_data(
            inhale_path="data/inhale.nii.gz",
            exhale_path="data/exhale.nii.gz",
            lung_mask_path="data/lung_mask.nii.gz",
            voxel_spacing=(1.5, 1.5, 3.0)  # mm
        )
    except FileNotFoundError:
        print("Sample data not found. Creating synthetic data for demonstration...")
        create_synthetic_data(pipeline)

    # Run the complete pipeline
    print("\n3. Running pipeline...")
    print("   - Image registration")
    print("   - Deformation analysis")
    print("   - Material property estimation")
    print("   - Ventilation analysis")

    # Run selected stages
    results = pipeline.run_pipeline(stages=[
        'registration',
        'deformation_analysis',
        'material_estimation',
        'ventilation_analysis'
    ])

    # Display results summary
    print("\n4. Results Summary:")
    summary = pipeline.get_summary()
    print(f"   Completed stages: {summary['completed_stages']}")
    print(f"   Data shape: {summary['data_shape']}")
    print(f"   Voxel spacing: {summary['voxel_spacing']} mm")

    # Get specific results
    ventilation_results = pipeline.get_results('ventilation')
    if ventilation_results:
        ventilation_map = ventilation_results['ventilation_map']
        print(f"   Ventilation map shape: {ventilation_map.shape}")
        print(f"   Ventilation statistics:")
        print(f"     Mean: {ventilation_map.mean():.4f}")
        print(f"     Std: {ventilation_map.std():.4f}")
        print(f"     Min: {ventilation_map.min():.4f}")
        print(f"     Max: {ventilation_map.max():.4f}")

    # Save results
    print("\n5. Saving results...")
    output_dir = Path("results/basic_pipeline_example")
    pipeline.save_results(output_dir, save_intermediate=True)
    print(f"   Results saved to: {output_dir}")

    print("\nPipeline completed successfully!")


def create_synthetic_data(pipeline):
    """Create synthetic data for demonstration purposes."""
    import numpy as np

    print("   Creating synthetic 4D-CT data...")

    # Create synthetic volume
    shape = (64, 64, 32)  # (D, H, W)
    lung_mask = np.zeros(shape, dtype=bool)

    # Create synthetic lung shape (ellipsoid)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center_z, center_y, center_x = shape[0] // 2, shape[1] // 2, shape[2] // 2
    radius_z, radius_y, radius_x = shape[0] // 3, shape[1] // 3, shape[2] // 3

    lung_mask = ((z - center_z) / radius_z) ** 2 + \
                ((y - center_y) / radius_y) ** 2 + \
                ((x - center_x) / radius_x) ** 2 <= 1

    # Create synthetic CT images (HU values)
    inhale_image = np.full(shape, -1000.0, dtype=np.float32)  # Air density
    exhale_image = np.full(shape, -1000.0, dtype=np.float32)

    # Add lung tissue density
    inhale_image[lung_mask] = np.random.normal(-850, 50, np.sum(lung_mask))
    exhale_image[lung_mask] = np.random.normal(-800, 50, np.sum(lung_mask))

    # Store synthetic data
    pipeline.data = {
        'inhale_image': inhale_image,
        'exhale_image': exhale_image,
        'lung_mask': lung_mask.astype(np.float32),
        'voxel_spacing': (1.5, 1.5, 3.0),
        'image_headers': {
            'inhale': {'voxel_size': (1.5, 1.5, 3.0)},
            'exhale': {'voxel_size': (1.5, 1.5, 3.0)}
        }
    }

    print(f"   Created synthetic data with shape {shape}")


def run_individual_components():
    """Run individual components for demonstration."""
    print("\n" + "=" * 40)
    print("Individual Components Example")
    print("=" * 40)

    # Create synthetic data
    import numpy as np
    shape = (32, 32, 16)
    lung_mask = np.ones(shape, dtype=bool)

    # Create synthetic displacement vector field
    dvf = np.random.randn(*shape, 3) * 2.0  # Small displacements

    print("\n1. Deformation Analysis Example:")
    deformation_analyzer = v4d.DeformationAnalyzer(gpu=False)  # Use CPU for example
    deformation_results = deformation_analyzer.analyze_deformation(
        dvf=dvf,
        voxel_spacing=(1.0, 1.0, 1.0),
        mask=lung_mask
    )

    print(f"   Deformation gradient shape: {deformation_results['deformation_gradient'].shape}")
    print(f"   Strain tensor shape: {deformation_results['strain_tensor'].shape}")

    print("\n2. Ventilation Calculation Example:")
    ventilation_calculator = v4d.VentilationCalculator()
    ventilation_results = ventilation_calculator.compute_ventilation(
        deformation_gradient=deformation_results['deformation_gradient'],
        lung_mask=lung_mask
    )

    ventilation_map = ventilation_results['ventilation_map']
    print(f"   Ventilation map shape: {ventilation_map.shape}")
    print(f"   Ventilation range: [{ventilation_map.min():.4f}, {ventilation_map.max():.4f}]")

    print("\n3. Material Modeling Example:")
    mechanical_modeler = v4d.MechanicalModeler()
    strain_tensor = deformation_results['strain_tensor']
    stress_results = mechanical_modeler.compute_stress(strain_tensor)

    print(f"   Stress tensor shape: {stress_results['stress_tensor'].shape}")
    print(f"   Von Mises stress range: [{stress_results['von_mises_stress'].min():.4f}, "
          f"{stress_results['von_mises_stress'].max():.4f}] kPa")


if __name__ == "__main__":
    setup_logging()

    try:
        # Run basic pipeline
        run_basic_pipeline()

        # Run individual components example
        run_individual_components()

    except Exception as e:
        print(f"\nError: {e}")
        print("This is a demonstration script. Make sure all dependencies are installed.")
        print("Install with: pip install -r requirements.txt")
        sys.exit(1)