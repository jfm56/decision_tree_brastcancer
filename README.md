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
- Clean, dependency-light codebase (no test, lint, or CI/CD files included)

## Algorithm Type
This project uses a **custom Decision Tree classifier** for binary classification (malignant vs. benign). The tree supports both entropy (information gain) and Gini impurity as split criteria. The implementation is fully from scratch (no scikit-learn tree code) and supports:
- Numeric features and thresholds
- Recursive tree building
- Pruning by max depth
- Flexible splitting criteria ('entropy' or 'gini')

## Structure
- `src/` - Source code (preprocessing, model, training)
- `data/` - Dataset location (not versioned)

## Workflow
1. **Data Preprocessing:** Cleans and bins features, handles missing values, and encodes labels.
2. **Model Training:** Trains a custom decision tree on the training set using entropy (default) or Gini impurity. Optionally performs grid search for max depth.
3. **Evaluation:** Computes accuracy, precision, recall, F1 score, and confusion matrix on the test set.
4. **Visualization:** Generates plots for the decision tree, feature importances, confusion matrix, and feature distributions.
5. **Reporting:** Compiles all results and visualizations into a single PDF report (`model_report.pdf`).

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

## Parameter Setup Code
You can configure the main model parameters in `src/train.py`:

```python
# src/train.py
from src.decision_tree import DecisionTreeClassifier

# Example parameter setup
clf = DecisionTreeClassifier(
    max_depth=3,              # Maximum tree depth
    criterion='entropy'       # 'entropy' or 'gini' for split criterion
)
clf.fit(x_train, y_train)
```

- **max_depth**: Maximum depth of the decision tree (default: 3)
- **criterion**: Splitting criterion, either 'entropy' (information gain) or 'gini'

You can modify these parameters directly in `src/train.py` to experiment with different settings.
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
