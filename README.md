<<<<<<< HEAD
# GLOBEMMobileNet
cs230 project
=======
# Neural Network Models for Depression Detection

This project implements and compares three neural network models for depression detection using behavioral and survey data. The models perform both binary classification (depression vs. no depression) and regression (predicting BDI scores).

## Dataset

The models are trained on the following datasets:
- **Training Data**: `11-10_train_dataset_W1.csv`
- **Validation Data**: `11-10_val_dataset_W1.csv`

### Features
- Input features derived from behavioral data (location, screen time, sleep, etc.)
- Target variable: BDI (Beck Depression Inventory) Total Score

## Models

### 1. Small NN Milestone (Baseline Model)
**Notebook**: `11-10_small-nn-milestone.ipynb`

**Architecture**:
- Input layer (10 features)
- 1 hidden layer with 5 units (ReLU activation)
- Dropout layer (0.5)
- Output layer:
  - Binary classification: 1 unit (sigmoid activation)
  - Regression: 1 unit (no activation)

**Training Configuration**:
- Optimizer: Adam (default learning rate = 0.001)
- Loss Functions:
  - Binary: `binary_crossentropy`
  - Regression: `mse`
- Epochs: 50
- Batch size: 32

**Prediction Files**:
- `binary_classification_predictions_small_nn_milestone.csv`
- `bdi_regression_predictions_small_nn_milestone.csv`

---

### 2. Small NN Low LR (Lower Learning Rate)
**Notebook**: `11-10_small-nn-low-lr.ipynb`

**Architecture**:
- Input layer (10 features)
- 1 hidden layer with 5 units (ReLU activation)
- Dropout layer (0.5)
- Output layer:
  - Binary classification: 1 unit (sigmoid activation)
  - Regression: 1 unit (no activation)

**Training Configuration**:
- Optimizer: Adam (learning rate = 0.0001)
- Loss Functions:
  - Binary: `binary_crossentropy`
  - Regression: `mse`
- Epochs: 50
- Batch size: 32

**Key Difference**: This model uses a 10x lower learning rate (0.0001 vs 0.001) compared to the baseline, which may lead to slower but potentially more stable convergence.

**Prediction Files**:
- `binary_classification_predictions_small_nn_low_lr.csv`
- `bdi_regression_predictions_small_nn_low_lr.csv`

---

### 3. Two Hidden Layers (Deeper Architecture)
**Notebook**: `11-10_two-layer-nn-low-lr.ipynb`

**Architecture**:
- Input layer (10 features)
- 1st hidden layer with 10 units (ReLU activation)
- Dropout layer (0.5)
- 2nd hidden layer with 10 units (ReLU activation)
- Dropout layer (0.5)
- Output layer:
  - Binary classification: 1 unit (sigmoid activation)
  - Regression: 1 unit (no activation)

**Training Configuration**:
- Optimizer: Adam (learning rate = 0.0001)
- Loss Functions:
  - Binary: `binary_crossentropy`
  - Regression: `mse`
- Epochs: 50
- Batch size: 32

**Key Difference**: This model has two hidden layers (vs. one) with more units per layer (10 vs. 5), providing greater model capacity to learn complex patterns.

**Prediction Files**:
- `binary_classification_predictions_two_layers.csv`
- `bdi_regression_predictions_two_layers.csv`

---

## Tasks

### Binary Classification
Predicts whether a participant has depression (binary outcome based on BDI score > 0).

**Evaluation Metrics**:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

### Regression (BDI Score Prediction)
Predicts the continuous BDI Total Score.

**Evaluation Metrics**:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## Model Comparison

To compare the performance of all three models, run the analysis notebook:

**Notebook**: `model_predictions_analysis.ipynb`

This notebook provides:
- Confusion matrices for binary classification
- Predicted vs. actual scatter plots for regression
- Residual analysis
- Side-by-side performance metrics
- Visual comparisons

**Generated Visualizations**:
- `binary_confusion_matrices.png` - Confusion matrices for all models
- `bdi_predicted_vs_actual.png` - Predicted vs. actual BDI scores
- `bdi_residuals.png` - Residual plots for regression models
- `model_comparison.png` - Bar chart comparison of model performance

---

## Requirements

```
tensorflow
keras
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Project Structure

```
.
├── 11-10_small-nn-milestone.ipynb           # Baseline model
├── 11-10_small-nn-low-lr.ipynb              # Lower learning rate model
├── 11-10_two-layer-nn-low-lr.ipynb          # Two hidden layers model
├── model_predictions_analysis.ipynb          # Model comparison and analysis
├── 11-10_train_dataset_W1.csv               # Training dataset (ignored)
├── 11-10_val_dataset_W1.csv                 # Validation dataset (ignored)
├── binary_classification_predictions_*.csv   # Binary predictions (ignored)
├── bdi_regression_predictions_*.csv          # Regression predictions (ignored)
├── *.png                                     # Generated visualizations
└── README.md                                 # This file
```

---

## Usage

1. **Train a model**: Open and run one of the model notebooks (e.g., `11-10_small-nn-milestone.ipynb`)
2. **Analyze results**: Run `model_predictions_analysis.ipynb` to compare all models
3. **View visualizations**: Check the generated PNG files for visual comparisons

---

## Model Selection Considerations

- **Small NN Milestone**: Good baseline with fast training
- **Small NN Low LR**: Use when baseline overfits or training is unstable
- **Two Hidden Layers**: Use when more model capacity is needed to capture complex patterns

Choose the model based on validation performance metrics and the specific requirements of your depression detection task.
>>>>>>> onad-dev
