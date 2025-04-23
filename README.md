# Decision Tree Classification Using Entropy for Breast Cancer Diagnosis

This project implements a custom Decision Tree classifier to predict breast cancer diagnoses (malignant or benign) using entropy (information gain) for optimal splits. The dataset features are binned into 'low', 'medium', and 'high' categories to improve interpretability. The codebase is modular, tested, and CI/CD-enabled.

## Features
- Custom entropy-based Decision Tree implementation
- Data preprocessing with feature binning
- Automated testing (Pytest)
- Code linting (Pylint)
- Test coverage (Coverage.py)
- CI/CD with GitHub Actions
- Reproducible, modular code

## Structure
- `src/` - Source code (preprocessing, model, training)
- `tests/` - Unit tests
- `data/` - Dataset location (not versioned)
- `.github/workflows/` - CI/CD workflows

## Setup
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
See the main scripts in `src/` for training and evaluation.
