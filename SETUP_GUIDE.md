# LMT Dimensionality Reduction Toolkit - Setup Guide

## 🚀 Quick Start

### Prerequisites Checklist
- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] Basic command line knowledge
- [ ] SQLite database from LMT experiments
- [ ] DB Browser for SQLite (optional but recommended)

## 📥 Installation Steps

### 1. Get the Code
```bash
# Clone the repository
git clone https://github.com/your-username/LMT-dim-reduction-toolkit
cd LMT-dim-reduction-toolkit
```

### 2a. Set Up Python Environment
```bash
# Create a virtual environment (recommended)
python -m venv lmt_env

# Activate the environment
# On Windows:
.\lmt_env\Scripts\activate
# On Unix/MacOS:
source lmt_env/bin/activate
```
### 2b. Set Up Conda Enviroment
```terminal
# Create a conda enviroment
conda -n lmt_env

# Activate the enviroment
conda activate lmt_env
```

### 3. Install Dependencies
⚠️ **Important**: Follow this exact order
```bash
# 1. First, install numpy
pip install numpy==1.23.5

# 2. Then scipy
pip install scipy>=1.9.0

# 3. Finally, install the package
pip install -e .  # for development mode (recommended)
# OR
pip install .    # for regular installation
```

After installation, you can import the package in Python:
```python
# Example imports
from lmt_analysis.preprocessing import event_filtered
from lmt_analysis.behavior import behavior_processor
from lmt_analysis.visualization import identity_space_plotter
```

### 4. Verify Installation
```python
# Create a test script (test_installation.py)
from lmt_analysis.utils import test_setup
test_setup.run_test()
```

If you see no errors, the installation was successful! 
















Best approach might be to:
Use median for the bar width (most representative)
Add std as error bars or transparency
Keep mean in the statistics panel
This way you can see:
Typical duration (median - width)
Consistency (std - error bars)
Overall pattern (mean - stats)