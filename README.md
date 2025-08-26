# ML-Playground-app

# Advanced ML Playground

🌟 An interactive **Streamlit** playground for exploring **classification, regression, and clustering algorithms** with visualization, cross-validation, and hyperparameter tuning.

---

## Features

1. **Data Options**
   - Built-in datasets: Iris, Wine, Breast Cancer, Moons, Circles, Blobs, Synthetic 3D
   - Upload your own CSV

2. **Algorithms**
   - Classification: Decision Tree, Random Forest, Logistic Regression, Bagging, AdaBoost, Gradient Boosting, Voting Classifier, XGBoost
   - Regression: Linear Regression, Gradient Boosting Regressor, XGBoost Regressor
   - Clustering: KMeans, DBSCAN

3. **Hyperparameter Tuning**
   - Slider inputs for key hyperparameters
   - Optional **GridSearchCV** to find best parameters

4. **Model Evaluation**
   - Train/test split with custom test size
   - Cross-validation (CV) with fold selection
   - CV score plots
   - Accuracy, classification report, RMSE, R² (depending on task)
   - Feature importance visualization (tree-based models)

5. **Visualization**
   - **2D decision boundary** for 2-feature datasets
   - **3D scatter** for 3-feature datasets
   - Cluster plots for clustering algorithms

6. **Model Export / Import**
   - Export trained model as `.pkl`
   - Upload `.pkl` model to reuse and test

7. **User-Friendly**
   - Sidebar configuration
   - Interactive plotting
   - Easy to understand metrics and hyperparameters

---

## Setup & Run

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-folder>
