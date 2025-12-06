# Complete Setup Guide 🔧

Step-by-step instructions for setting up the project from scratch.

## System Requirements

### Minimum
- 4GB RAM
- 2GB free disk space
- Python 3.8+

### Recommended
- 8GB+ RAM (for faster training)
- 10GB free disk space (for models & data)
- GPU (NVIDIA) for 10x faster training

## Python Installation

### Windows

1. **Download Python**
   - Visit: https://www.python.org/downloads/
   - Download "Python 3.11" installer

2. **Install Python**
   - Run installer
   - ✅ CHECK: "Add Python to PATH"
   - Click "Install Now"

3. **Verify Installation**
   ```bash
   python --version
   # Should show: Python 3.11.x
   ```

### macOS

```bash
brew install python3
python3 --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

## Virtual Environment Setup

### Windows

```bash
python -m venv venv
venv\Scripts\activate
# You should see (venv) prefix in terminal
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
# You should see (venv) prefix in terminal
```

## Project Setup

### 1. Create Project Directory

```bash
mkdir trendmaster
cd trendmaster
```

### 2. Copy All Files

Copy these files into the folder:
- requirements.txt
- config.py
- technical_indicators.py
- data_processor.py
- model_trainer.py
- main.py
- All .md documentation files

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import tensorflow; import streamlit; print('✓ All packages OK')"
```

### 5. Create Directories

```bash
# Directories auto-create on first run, but can create manually:
mkdir data models predictions
mkdir models/scalers
```

## Next Steps

1. ✅ Verify all packages installed
2. ✅ Run `streamlit run main.py`
3. ✅ Follow QUICKSTART.md for first prediction

---

**You're all set!** 🚀