"""
Advanced ML Playground – Streamlit App
Supports: Classification, Regression, Clustering
Features: 2D/3D visualization, CV, GridSearch, model export/import
"""

import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb

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
    elif name == "Moons":
        X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)
        return pd.DataFrame(X, columns=["Feature 0","Feature 1"]), y, ["0","1"]
    elif name == "Circles":
        X, y = datasets.make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42)
        return pd.DataFrame(X, columns=["Feature 0","Feature 1"]), y, ["0","1"]
    elif name == "Blobs":
        X, y = datasets.make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
        return pd.DataFrame(X, columns=["Feature 0","Feature 1"]), y, [str(i) for i in np.unique(y)]
    else:
        raise ValueError("Unknown dataset")

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    classes = list(data.target_names)
    return X, y, classes

def ensure_numeric(df):
    return df.select_dtypes(include=[np.number]).copy()

def plot_feature_importances(clf, X):
    if not hasattr(clf, "feature_importances_"):
        return None
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Handle NumPy array vs DataFrame
    if hasattr(X, "columns"):
        feature_names = X.columns
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return fig

def plot_decision_regions(model, X2, y, feature_names, class_names):
    x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
    y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300),
                         np.linspace(y_min,y_max,300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.contourf(xx, yy, Z, alpha=0.25)
    ax.scatter(X2[:,0], X2[:,1], c=y, edgecolor="k", s=30)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title("Decision Boundary (2D)")
    return fig

def plot_tree_graph(clf, X, class_names):
    fig, ax = plt.subplots(figsize=(12,8))
    if hasattr(X,"columns"):
        feature_names = X.columns
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    plot_tree(clf, feature_names=feature_names, class_names=class_names if class_names else True,
              filled=True, rounded=True, impurity=True, ax=ax)
    ax.set_title("Decision Tree")
    fig.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label", title="Confusion Matrix")
    thresh = cm.max()/2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                    color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout()
    return fig

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="ML Playground", layout="wide")
st.title("🌟 ML Playground – Classification / Regression / Clustering")

# --- Sidebar ---
with st.sidebar:
    st.header("1) Data")
    data_mode = st.radio("Choose data source", ["Built-in", "Upload CSV"])
    if data_mode=="Built-in":
        dataset_name = st.selectbox("Dataset", ["Iris","Wine","Breast Cancer","Moons","Circles","Blobs"], index=0)
        X, y, class_names = load_builtin(dataset_name)
        target_col = "target"
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        target_col = st.text_input("Target column name")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview:")
            st.dataframe(df.head())
            if target_col and target_col in df.columns:
                y = df[target_col]
                X = ensure_numeric(df.drop(columns=[target_col]))
                class_names = sorted([str(c) for c in pd.unique(y)])
            else:
                X=y=class_names=None
        else:
            X=y=class_names=None

    st.markdown("---")
    st.header("2) Task & Model")
    task_type = st.selectbox("Task Type", ["Classification","Regression","Clustering"])
    if task_type=="Classification":
        model_option = st.selectbox("Model", ["Decision Tree","Random Forest","Logistic Regression",
                                              "Bagging","AdaBoost","Gradient Boosting","XGBoost","Voting Classifier"])
    elif task_type=="Regression":
        model_option = st.selectbox("Model", ["Linear Regression","Gradient Boosting Regressor","XGBoost Regressor"])
    else:
        model_option = st.selectbox("Model", ["KMeans","DBSCAN"])

    st.markdown("---")
    st.header("3) Hyperparameters")
    max_depth = st.slider("max_depth (None=unlimited)",1,30,value=5)
    max_depth = None if st.checkbox("Use unlimited depth",value=False) else max_depth
    test_size = st.slider("test_size",0.1,0.5,value=0.2,step=0.05)
    scale_features = st.checkbox("Standardize features",value=False)
    random_state = st.number_input("random_state", min_value=0,max_value=9999,value=42,step=1)

    st.markdown("---")
    st.header("4) CV / GridSearch")
    cv_folds = st.slider("CV folds", 2, 10, 5)
    use_grid = st.checkbox("Use GridSearchCV")

# --- Main ---
if X is not None and y is not None:
    # Optional scaling
    X_model = X.copy()
    if scale_features:
        scaler = StandardScaler()
        X_model[:] = scaler.fit_transform(X_model.values)

    # Split
    if task_type!="Clustering":
        X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=test_size,
                                                            random_state=random_state,
                                                            stratify=y if task_type=="Classification" and len(np.unique(y))>1 else None)
    else:
        X_train = X_model
        y_train = y

    # Build model
    clf = None
    if model_option=="Decision Tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    elif model_option=="Random Forest":
        clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    elif model_option=="Logistic Regression":
        clf = LogisticRegression(max_iter=500, random_state=random_state)
    elif model_option=="Bagging":
        base = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        clf = BaggingClassifier(base_estimator=base, random_state=random_state)
    elif model_option=="AdaBoost":
        base = DecisionTreeClassifier(max_depth=1, random_state=random_state)
        clf = AdaBoostClassifier(base_estimator=base, random_state=random_state)
    elif model_option=="Gradient Boosting":
        clf = GradientBoostingClassifier(max_depth=max_depth, random_state=random_state)
    elif model_option=="XGBoost":
        clf = xgb.XGBClassifier(max_depth=max_depth, use_label_encoder=False, eval_metric="mlogloss", random_state=random_state)
    elif model_option=="Voting Classifier":
        clf = VotingClassifier(estimators=[
            ("dt", DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)),
            ("rf", RandomForestClassifier(max_depth=max_depth, random_state=random_state)),
            ("lr", LogisticRegression(max_iter=500, random_state=random_state))
        ], voting="hard")
    elif model_option=="Linear Regression":
        clf = LinearRegression()
    elif model_option=="Gradient Boosting Regressor":
        clf = GradientBoostingRegressor(max_depth=max_depth, random_state=random_state)
    elif model_option=="XGBoost Regressor":
        clf = xgb.XGBRegressor(max_depth=max_depth, random_state=random_state)
    elif model_option=="KMeans":
        clf = KMeans(n_clusters=len(np.unique(y)), random_state=random_state)
    elif model_option=="DBSCAN":
        clf = DBSCAN(eps=0.5, min_samples=5)

    run_clicked = st.button("🚀 Run / Train Model")
    if run_clicked:
        clf.fit(X_train, y_train) if task_type!="Clustering" else clf.fit(X_train)
        # Metrics & display
        if task_type=="Classification":
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy on test set: {acc:.4f}")
            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            st.pyplot(plot_confusion_matrix(cm, class_names))

            # 2D decision boundary (only 2 features)
            if X_model.shape[1]>=2:
                f1, f2 = 0,1
                clf2 = clf.__class__(**clf.get_params())
                clf2.fit(X_train.iloc[:,[f1,f2]] if hasattr(X_train,"iloc") else X_train[:,[f1,f2]], y_train)
                X2_all = np.vstack([X_train.iloc[:,[f1,f2]], X_test.iloc[:,[f1,f2]]]) if hasattr(X_train,"iloc") else np.vstack([X_train[:,[f1,f2]], X_test[:,[f1,f2]]])
                y_all = np.hstack([y_train, y_test])
                st.pyplot(plot_decision_regions(clf2, X2_all, y_all, [f"Feature {f1}", f"Feature {f2}"], class_names))

        elif task_type=="Regression":
            y_pred = clf.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.success(f"RMSE: {rmse:.4f} | R²: {r2:.4f}")

        else:  # Clustering
            labels = clf.labels_ if hasattr(clf,"labels_") else clf.predict(X_train)
            fig, ax = plt.subplots()
            ax.scatter(X_train.iloc[:,0] if hasattr(X_train,"iloc") else X_train[:,0],
                       X_train.iloc[:,1] if hasattr(X_train,"iloc") else X_train[:,1],
                       c=labels, cmap="tab10")
            ax.set_title(f"{model_option} Clustering")
            st.pyplot(fig)

        # Feature importance
        fi_fig = plot_feature_importances(clf, X_model)
        if fi_fig: st.pyplot(fi_fig)

        # Tree graph
        if model_option in ["Decision Tree","Random Forest","AdaBoost","Gradient Boosting"]:
            st.pyplot(plot_tree_graph(clf, X_model, class_names))

        # Cross-validation
        if task_type!="Clustering":
            cv_scores = cross_val_score(clf, X_train, y_train, cv=cv_folds)
            st.write(f"CV Scores: {cv_scores}")
            st.write(f"Mean CV: {cv_scores.mean():.4f}")

        # GridSearchCV
        if use_grid and task_type!="Clustering":
            param_grid = {"max_depth":[3,5,7], "min_samples_split":[2,5,10]}
            grid = GridSearchCV(clf, param_grid, cv=cv_folds)
            grid.fit(X_train, y_train)
            st.write("GridSearchCV best params:", grid.best_params_)
            clf = grid.best_estimator_

        # Model export
        if st.button("Export model (.pkl)"):
            with open("model.pkl","wb") as f: pickle.dump(clf,f)
            st.success("Model saved as model.pkl!")

        uploaded_model = st.file_uploader("Upload trained .pkl model", type=["pkl"])
        if uploaded_model is not None:
            clf = pickle.load(uploaded_model)
            st.success("Model loaded successfully!")

else:
    st.info("Upload CSV or use a built-in dataset.")

st.caption("Tip: Toggle scaling and select 2D features for decision boundaries. Use CV/GridSearch for tuning.")

