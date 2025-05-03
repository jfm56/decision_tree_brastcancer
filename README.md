# IS665-852, Group 1, Project 2 Data Mining

**Author:** James Mullen

This project implements a custom Decision Tree classifier to predict breast cancer diagnoses (malignant or benign) using either entropy (information gain) or Gini impurity for optimal splits. The pipeline produces a comprehensive, print-friendly PDF report with all key visualizations, summary statistics, and model evaluation metrics.

## Features
- Custom Decision Tree implementation (supports 'entropy' or 'gini' splitting)
- Data preprocessing with feature binning
- Cross-validated grid search for optimal tree depth
- Print-friendly, multi-page PDF report (all results in one file)
- Visualizations: tree, confusion matrix, feature importance, distributions, correlation heatmap
- Summary statistics and confusion-matrix metrics
- Automated testing (Pytest), code linting (Pylint), CI/CD

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

### 1. Prepare your environment
```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Train and generate report
Run from the project root:
```sh
python3 -m src.train
```

### 3. View your results
- Open `model_report.pdf` for a print-ready summary of all visualizations, metrics, and statistics.
- The PDF includes:
  - Cover page with project, author, and date
  - Summary statistics for top features
  - Confusion matrix and metrics table
  - Decision tree visualization
  - Feature importance plot
  - Top feature distributions (KDE, box, binned)
  - Correlation heatmap

### 4. Customization
- To use Gini or entropy, set `criterion='gini'` or `criterion='entropy'` in `DecisionTreeClassifier` in `src/train.py`.
- To change the number of top features or bins, edit the arguments in the call to `save_all_visualizations_pdf`.

---

For questions or further customization, contact the author or open an issue.
