"""
Setup script for Vent4D-Mech
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="vent4d-mech",
    version="0.1.0",
    author="Vent4D-Mech Development Team",
    author_email="vent4d-mech@example.com",
    description="Python-based Lung Tissue Dynamics Modeling Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/vent4d-mech",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vent4d-mech=vent4d_mech.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "vent4d_mech": [
            "config/*.yaml",
        ],
    },
    zip_safe=False,
)