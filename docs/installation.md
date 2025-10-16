# AITA Installation Guide

This guide covers different ways to install AITA dependencies based on your needs.

## Prerequisites

- Python 3.8 or higher
- Git (for development)
- Google Cloud Storage account with service account credentials

## Installation Options

### Option 1: Full Installation (Recommended)

Install all dependencies for complete functionality:

```bash
# Clone or navigate to the AITA directory
cd /path/to/AITA

# Install all dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Option 2: Development Installation

For developers who want to contribute or modify AITA:

```bash
# Install development dependencies (includes testing, linting, etc.)
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Option 3: Minimal Installation

For constrained environments or when you only need core functionality:

```bash
# Install minimal dependencies
pip install -r requirements-minimal.txt

# Note: You may need additional packages for full EasyOCR functionality
```

### Option 4: Direct Package Installation

Install from pyproject.toml:

```bash
# Install the package directly
pip install -e .

# Install with development extras
pip install -e ".[dev]"
```

## Environment Setup

### 1. Create Environment File

Copy the example environment file and update it with your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual values:

```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-2.5-flash

# Google Cloud Storage Configuration
GOOGLE_CLOUD_PROJECT_ID=vital-orb-340815
GOOGLE_CLOUD_STORAGE_BUCKET=exam1_uwmadison
GOOGLE_APPLICATION_CREDENTIALS=C:\Users\ellio\OneDrive - UW-Madison\AITA\vital-orb-340815-7a97e5eea09e.json

# Application Configuration
AITA_DATA_DIR=./data
AITA_INTERMEDIATE_DIR=C:\Users\ellio\OneDrive - UW-Madison\AITA\intermediateproduct
```

### 2. Create Required Directories

```bash
# Create data directories
python -c "from aita.config import get_config; get_config().ensure_data_directories()"
```

## Verification

### Test Your Installation

```bash
# Run the integration test
python test_integration.py

# Run unit tests (if you have dev dependencies)
pytest

# Test individual components
python examples/test_llm_demo.py
python examples/test_storage_demo.py
```

### Quick Verification Script

```python
# test_install.py
try:
    from aita.services.llm.openrouter import create_openrouter_client
    from aita.services.storage import create_storage_service
    from aita.config import get_config

    print("✅ All imports successful!")

    # Test configuration
    config = get_config()
    print(f"✅ Configuration loaded: {config.google_cloud.project_id}")

    print("🎉 Installation appears successful!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run: pip install -r requirements.txt")

except Exception as e:
    print(f"⚠️  Configuration issue: {e}")
    print("Check your .env file settings")
```

## Troubleshooting

### Common Issues

#### 1. EasyOCR Installation Problems

```bash
# If EasyOCR fails to install, try:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install easyocr
```

#### 2. Google Cloud Authentication

```bash
# Verify your credentials file exists
ls -la "C:\Users\ellio\OneDrive - UW-Madison\AITA\vital-orb-340815-7a97e5eea09e.json"

# Test authentication
python -c "from google.cloud import storage; client = storage.Client(); print('GCS auth successful')"
```

#### 3. OpenRouter API Issues

```bash
# Test your API key
python -c "
import os
from aita.services.llm.openrouter import create_openrouter_client
client = create_openrouter_client()
print('LLM client created successfully')
"
```

#### 4. Memory Issues with Large Models

If you encounter memory issues with EasyOCR or torch:

```bash
# Install CPU-only versions
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or use the minimal installation
pip install -r requirements-minimal.txt
```

### Platform-Specific Notes

#### Windows
- Ensure you have Visual C++ Build Tools installed for some packages
- Use PowerShell or Command Prompt with Administrator privileges if needed

#### macOS
- You may need to install Xcode Command Line Tools: `xcode-select --install`

#### Linux
- Ensure you have the necessary system libraries for OpenCV and image processing

## Virtual Environment (Recommended)

Always use a virtual environment to avoid conflicts:

```bash
# Create virtual environment
python -m venv aita-env

# Activate it
# On Windows:
aita-env\Scripts\activate
# On macOS/Linux:
source aita-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Next Steps

After successful installation:

1. **Configure your environment** (`.env` file)
2. **Test the integration** with `python test_integration.py`
3. **Run the demos** in the `examples/` directory
4. **Start using AITA** for your exam grading workflow

For detailed usage instructions, see:
- `docs/llm_usage_examples.md` - LLM service examples
- `docs/storage_service_guide.md` - Storage service guide
- `examples/` directory - Working examples and demos