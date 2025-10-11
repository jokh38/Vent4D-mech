# Vent4D-Mech Development Commands

## Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Install GPU support (optional but recommended)
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

## Development Workflow
```bash
# Run basic pipeline example
python examples/basic_pipeline.py

# Use CLI entry point
vent4d-mech --help
vent4d-mech --config config/default.yaml --data data/
```

## Testing Commands
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/test_registration.py
pytest tests/test_deformation.py
pytest tests/test_mechanical.py

# Run with coverage
pytest --cov=vent4d_mech

# Run with verbose output
pytest -v

# Run specific test pattern
pytest -k "test_registration" -v
```

## Code Quality Commands
```bash
# Format code with Black
black vent4d_mech/ tests/

# Lint code with flake8
flake8 vent4d_mech/ tests/

# Type checking with mypy
mypy vent4d_mech/

# Run all quality checks together
black vent4d_mech/ tests/ && flake8 vent4d_mech/ tests/ && mypy vent4d_mech/
```

## Documentation Commands
```bash
# Build documentation (if docs/ exists)
cd docs/
make html

# Clean build
make clean

# Serve documentation locally
python -m http.server 8000 -d _build/html/
```

## Git Commands
```bash
# Check git status
git status

# Add and commit changes
git add .
git commit -m "Descriptive commit message"

# Push changes
git push

# Check recent commits
git log --oneline -10

# Create new branch
git checkout -b feature/new-feature
```

## System Utility Commands
```bash
# List files in directory
ls -la

# Find files by pattern
find . -name "*.py" -type f

# Search for text in files
grep -r "search_term" vent4d_mech/

# Check Python version
python --version

# Check installed packages
pip list

# Check available GPU (Linux)
nvidia-smi

# Monitor system resources
htop  # or top

# Check disk usage
df -h

# Check memory usage
free -h
```

## Performance Monitoring
```bash
# Profile Python code
python -m cProfile -o profile.stats your_script.py

# Memory profiling
pip install memory_profiler
python -m memory_profiler your_script.py

# Line profiling
pip install line_profiler
kernprof -l -v your_script.py
```

## Docker Commands (if Dockerized)
```bash
# Build Docker image
docker build -t vent4d-mech .

# Run container
docker run -it --rm vent4d-mech

# Run with GPU support
docker run --gpus all -it --rm vent4d-mech
```

## Data Management
```bash
# Create synthetic test data (from example)
python examples/basic_pipeline.py

# Convert DICOM to NIfTI (if needed)
python -c "import nibabel as nib; import SimpleITK as sitk; # conversion code"

# Check data file integrity
python -c "import numpy as np; data = np.load('data.npy'); print(data.shape)"
```

## Configuration Management
```bash
# Validate configuration
python -c "import yaml; config = yaml.safe_load(open('config/default.yaml')); print(config.keys())"

# Create custom config
cp config/default.yaml config/custom.yaml
# Edit config/custom.yaml as needed
```