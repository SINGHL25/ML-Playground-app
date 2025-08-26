"""
ML Playground – Streamlit App
Features:
• Multiple classifiers/regressors
• Built-in datasets or CSV upload
• 2D & 3D decision boundaries
• Hyperparameter tuning, GridSearchCV
• Cross-validation
• Model export/import (pickle)
• Feature importances and metrics
• Explanations for each model
"""
import io, pickle, textwrap
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, r2_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN

# Optional: XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# ----------------------------
# Utility functions
# ----------------------------
def load_builtin(name: str):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Wine":
        data = datasets.load_wine()
    elif name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        raise ValueError("Unknown dataset")
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    classes = list(data.target_names)
    return X, y, classes

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).copy()

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i,j],'d'),
                    ha="center", va="center",
                    color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout()
    return fig

def plot_tree_graph(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names if class_names else True,
              filled=True, rounded=True, impurity=True, ax=ax)
    fig.tight_layout()
    return fig

def plot_feature_importances(clf, feature_names):
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return fig

def plot_decision_regions(model, X2, y, feature_names, class_names):
    x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
    y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.contourf(xx,yy,Z,alpha=0.25)
    ax.scatter(X2[:,0], X2[:,1], c=y, edgecolor="k", s=30)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Decision Boundary (2D)")
    return fig

def plot_3d_scatter(X, y, title="3D Visualization", is_classification=True):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    X_vals = X.values if hasattr(X,"values") else X
    if is_classification:
        ax.scatter(X_vals[:,0], X_vals[:,1], X_vals[:,2], c=y, cmap="tab10", s=40, edgecolor="k")
    else:
        p = ax.scatter(X_vals[:,0], X_vals[:,1], X_vals[:,2], c=y, cmap="viridis", s=40)
        fig.colorbar(p, ax=ax, shrink=0.6)
    ax.set_xlabel("Feature 0"); ax.set_ylabel("Feature 1"); ax.set_zlabel("Feature 2")
    ax.set_title(title)
    return fig

def plot_3d_decision_boundary(model, X3, y, title="3D Decision Boundary"):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    x_min,x_max = X3[:,0].min()-1, X3[:,0].max()+1
    y_min,y_max = X3[:,1].min()-1, X3[:,1].max()+1
    z_min,z_max = X3[:,2].min()-1, X3[:,2].max()+1
    xx,yy,zz = np.meshgrid(np.linspace(x_min,x_max,30),
                            np.linspace(y_min,y_max,30),
                            np.linspace(z_min,z_max,30))
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    mask = np.random.rand(*Z.shape)<0.05
    ax.scatter(xx[mask], yy[mask], zz[mask], c=Z[mask], cmap="tab10", alpha=0.2, s=20)
    ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=y, cmap="tab10", edgecolor="k", s=40)
    ax.set_xlabel("Feature 0"); ax.set_ylabel("Feature 1"); ax.set_zlabel("Feature 2")
    ax.set_title(title)
    return fig

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("🤖 ML Playground – Classification / Regression / Clustering")

with st.sidebar:
    st.header("1) Data")
    data_mode = st.radio("Choose data source", ["Built-in", "Upload CSV"], horizontal=True)
    if data_mode=="Built-in":
        dataset_name = st.selectbox("Dataset", ["Iris","Wine","Breast Cancer"], index=0)
        X, y, class_names = load_builtin(dataset_name)
        target_col = "target"
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        target_col = st.text_input("Target column")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:"); st.dataframe(df.head())
            if target_col and target_col in df.columns:
                y = df[target_col]; X = df.drop(columns=[target_col])
                X = ensure_numeric(X)
                class_names = sorted([str(c) for c in pd.unique(y)])
            else: X=y=class_names=None
        else: X=y=class_names=None

    st.markdown("---")
    st.header("2) Model Selection")
    model_option = st.selectbox("Choose model", [
        "Decision Tree", "Random Forest", "Logistic Regression",
        "Linear Regression", "Bagging", "AdaBoost", "Gradient Boosting",
        "Voting Classifier", "KMeans", "DBSCAN", "XGBoost" if xgb_available else "XGBoost (not installed)"
    ])
    st.markdown("**Explanations**:")
    if model_option=="Decision Tree":
        st.write("Decision Tree splits data by feature thresholds to classify.")
    elif model_option=="Random Forest":
        st.write("Random Forest is an ensemble of Decision Trees for better generalization.")
    elif model_option=="Logistic Regression":
        st.write("Logistic Regression predicts probability of classes using sigmoid function.")
    elif model_option=="Linear Regression":
        st.write("Linear Regression fits a line to predict continuous values.")
    elif model_option=="Bagging":
        st.write("Bagging trains multiple base models on bootstrapped samples and averages predictions.")
    elif model_option=="AdaBoost":
        st.write("AdaBoost combines weak classifiers sequentially, focusing on errors.")
    elif model_option=="Gradient Boosting":
        st.write("Gradient Boosting fits models sequentially to minimize loss using gradients.")
    elif model_option=="Voting Classifier":
        st.write("Voting Classifier combines multiple models by majority vote (classification) or average (regression).")
    elif model_option=="KMeans":
        st.write("KMeans clustering groups data into k clusters by minimizing distance to centroids.")
    elif model_option=="DBSCAN":
        st.write("DBSCAN clustering groups dense regions and marks noise points.")
    elif model_option.startswith("XGBoost"):
        st.write("XGBoost: Gradient boosting with regularization and optimized performance.")

    st.markdown("---")
    st.header("3) Train/Test & Hyperparameters")
    test_size = st.slider("Test size", 0.1, 0.5, value=0.2, step=0.05)
    scale_features = st.checkbox("Standardize features", value=False)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

st.markdown("---")
# ----------------------------
# Model Training & Playground
# ----------------------------
if X is not None and y is not None:
    X_model = X.copy()
    if scale_features:
        scaler = StandardScaler()
        X_model[:] = scaler.fit_transform(X_model.values)

    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size, random_state=random_state,
                                                        stratify=y if len(pd.unique(y))>1 else None)

    # Define model with default hyperparameters
    if model_option=="Decision Tree":
        clf = DecisionTreeClassifier(random_state=random_state)
        task_type = "Classification"
    elif model_option=="Random Forest":
        clf = RandomForestClassifier(random_state=random_state)
        task_type = "Classification"
    elif model_option=="Logistic Regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        task_type = "Classification"
    elif model_option=="Linear Regression":
        clf = LinearRegression()
        task_type = "Regression"
    elif model_option=="Bagging":
        clf = BaggingClassifier(random_state=random_state)
        task_type = "Classification"
    elif model_option=="AdaBoost":
        clf = AdaBoostClassifier(random_state=random_state)
        task_type = "Classification"
    elif model_option=="Gradient Boosting":
        clf = GradientBoostingClassifier(random_state=random_state)
        task_type = "Classification"
    elif model_option=="Voting Classifier":
        clf = VotingClassifier(estimators=[('dt',DecisionTreeClassifier()),('rf',RandomForestClassifier()),('lr',LogisticRegression(max_iter=1000))])
        task_type = "Classification"
    elif model_option=="KMeans":
        clf = KMeans(n_clusters=len(pd.unique(y)), random_state=random_state)
        task_type = "Clustering"
    elif model_option=="DBSCAN":
        clf = DBSCAN()
        task_type = "Clustering"
    elif model_option=="XGBoost":
        clf = XGBClassifier(random_state=random_state) if len(pd.unique(y))>1 else XGBRegressor(random_state=random_state)
        task_type = "Classification" if len(pd.unique(y))>1 else "Regression"
    else:
        st.error("Model not implemented.")
        clf=None

    run_clicked = st.button("🚀 Run / Train Model")
    if run_clicked and clf is not None:
        clf.fit(X_train, y_train)
        st.success("Model trained!")

        # Metrics
        if task_type=="Classification":
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc:.4f}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            st.pyplot(plot_confusion_matrix(cm, class_names=sorted([str(c) for c in pd.unique(y)])))

        elif task_type=="Regression":
            y_pred = clf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"R2 Score: {r2:.4f}, MSE: {mse:.4f}")

        # 2D Decision Boundary (if >=2 features)
        if X_model.shape[1]>=2 and task_type=="Classification":
            f1,f2 = X_model.columns[0], X_model.columns[1]
            clf2 = clf.__class__()
            clf2.fit(X_train[[f1,f2]], y_train)
            st.pyplot(plot_decision_regions(clf2, X_train[[f1,f2]].values, y_train.values, [f1,f2], sorted([str(c) for c in pd.unique(y)])))

        # 3D Scatter + Decision Boundary (if >=3 features)
        if X_model.shape[1]>=3:
            st.subheader("3D Scatter (first 3 features)")
            X3 = X_model.iloc[:, :3]
            st.pyplot(plot_3d_scatter(X3, y_train, is_classification=(task_type=="Classification")))
            if task_type=="Classification":
                clf3d = clf.__class__(); clf3d.fit(X_train.iloc[:,:3], y_train)
                st.pyplot(plot_3d_decision_boundary(clf3d, X_train.iloc[:,:3].values, y_train.values))

        # Feature importances
        fi_fig = plot_feature_importances(clf, X_model.columns if hasattr(X_model,"columns") else list(range(X_model.shape[1])))
        if fi_fig: st.pyplot(fi_fig)

        # Cross-validation
        if task_type in ["Classification","Regression"]:
            cv_scores = cross_val_score(clf, X_model, y, cv=5)
            st.write(f"5-Fold CV Scores: {cv_scores}")
            st.write(f"Mean CV Score: {cv_scores.mean():.4f}")

        # Model export/import
        st.subheader("Export / Import Model")
        export_file = st.button("Export Model (Pickle)")
        if export_file:
            with open("model.pkl","wb") as f:
                pickle.dump(clf,f)
            st.success("Model exported as model.pkl")

        uploaded_model = st.file_uploader("Upload Pickle Model", type=["pkl"])
        if uploaded_model:
            clf_loaded = pickle.load(uploaded_model)
            st.success("Model imported!")


