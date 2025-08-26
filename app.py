import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification, make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Classifiers / Regressors / Clustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb

st.set_page_config(page_title="Advanced ML Playground", layout="wide")
st.title("🧩 Advanced ML Playground (Classification / Regression / Clustering)")

# ---------------------------
# Dataset
dataset_type = st.sidebar.selectbox("Dataset Type", ["Built-in", "Upload CSV"])
if dataset_type == "Built-in":
    dataset_name = st.sidebar.selectbox("Dataset", ["Iris", "Wine", "Breast Cancer", "Moons", "Circles", "Blobs", "Synthetic 3D"])
    if dataset_name == "Iris": data = load_iris(as_frame=True); X, y = data.data, data.target
    elif dataset_name == "Wine": data = load_wine(as_frame=True); X, y = data.data, data.target
    elif dataset_name == "Breast Cancer": data = load_breast_cancer(as_frame=True); X, y = data.data, data.target
    elif dataset_name == "Moons": X, y = make_moons(n_samples=500, noise=0.3)
    elif dataset_name == "Circles": X, y = make_circles(n_samples=500, noise=0.2, factor=0.5)
    elif dataset_name == "Blobs": X, y = make_blobs(n_samples=500, centers=2)
    else: X, y = make_classification(n_samples=500, n_features=3, n_classes=2, n_informative=3, n_redundant=0)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(df.head())
        target_col = st.sidebar.selectbox("Target Column", df.columns)
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        st.stop()

# ---------------------------
# Task Type
task = st.sidebar.selectbox("Task Type", ["Classification", "Regression", "Clustering"])

# ---------------------------
# Algorithm Selection
if task=="Classification":
    model_name = st.sidebar.selectbox("Model", ["Decision Tree","Random Forest","Logistic Regression","Bagging","AdaBoost","Gradient Boosting","Voting Classifier","XGBoost"])
elif task=="Regression":
    model_name = st.sidebar.selectbox("Model", ["Linear Regression","Gradient Boosting Regressor","XGBoost Regressor"])
else:
    model_name = st.sidebar.selectbox("Model", ["KMeans","DBSCAN"])

# ---------------------------
# Hyperparameters
params = {}
if model_name=="Decision Tree": params["max_depth"] = st.sidebar.slider("Max Depth",1,30,5)
elif model_name=="Random Forest": params["n_estimators"] = st.sidebar.slider("Trees",10,200,100)
elif model_name=="Logistic Regression": params["C"] = st.sidebar.slider("Regularization C",0.01,10.0,1.0)
elif model_name=="Bagging": params["n_estimators"] = st.sidebar.slider("Estimators",10,200,50)
elif model_name=="AdaBoost": params["n_estimators"] = st.sidebar.slider("Estimators",10,200,50)
elif model_name=="Gradient Boosting" or model_name=="Gradient Boosting Regressor": params["n_estimators"] = st.sidebar.slider("Estimators",50,200,100)
elif model_name=="XGBoost" or model_name=="XGBoost Regressor": params["n_estimators"] = st.sidebar.slider("Estimators",50,200,100)
elif model_name=="KMeans": params["n_clusters"] = st.sidebar.slider("Clusters",2,10,3)
elif model_name=="DBSCAN": params["eps"] = st.sidebar.slider("eps",0.1,5.0,0.5); params["min_samples"] = st.sidebar.slider("min_samples",1,20,5)

# ---------------------------
# Train/Test Split
test_size = st.sidebar.slider("Test Size",0.1,0.5,0.2)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)

# ---------------------------
# CV & GridSearch
st.sidebar.header("⚙️ Cross-Validation & GridSearch")
cv_folds = st.sidebar.slider("CV Folds",2,10,5)
use_grid = st.sidebar.checkbox("Use GridSearchCV")
param_grid = {}
if use_grid and model_name in ["Decision Tree","Random Forest","Gradient Boosting","XGBoost"]:
    if model_name=="Decision Tree": param_grid = {"max_depth":[3,5,7,None],"min_samples_split":[2,5,10]}
    elif model_name=="Random Forest": param_grid = {"n_estimators":[50,100,200],"max_depth":[3,5,7,None]}
    elif model_name=="Gradient Boosting" or model_name=="XGBoost": param_grid = {"n_estimators":[50,100],"learning_rate":[0.01,0.1,0.2]}

# ---------------------------
# Model Factory
def get_model(name,params):
    if name=="Decision Tree": return DecisionTreeClassifier(**params)
    elif name=="Random Forest": return RandomForestClassifier(**params)
    elif name=="Logistic Regression": return LogisticRegression(max_iter=500,**params)
    elif name=="Bagging": return BaggingClassifier(**params)
    elif name=="AdaBoost": return AdaBoostClassifier(**params)
    elif name=="Gradient Boosting": return GradientBoostingClassifier(**params)
    elif name=="Voting Classifier":
        dt = DecisionTreeClassifier(max_depth=3); lr = LogisticRegression(max_iter=500); rf = RandomForestClassifier(n_estimators=50)
        return VotingClassifier([("dt",dt),("lr",lr),("rf",rf)])
    elif name=="XGBoost": return xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', **params)
    elif name=="Linear Regression": return LinearRegression(**params)
    elif name=="Gradient Boosting Regressor": return GradientBoostingRegressor(**params)
    elif name=="XGBoost Regressor": return xgb.XGBRegressor(**params)
    elif name=="KMeans": return KMeans(**params)
    elif name=="DBSCAN": return DBSCAN(**params)
    return None

model = get_model(model_name,params)

# ---------------------------
# Run
if st.button("Train / Run Model"):
    if use_grid and param_grid:
        grid = GridSearchCV(model, param_grid, cv=cv_folds, n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        st.success(f"Best params: {grid.best_params_}")
        st.write(f"CV score: {grid.best_score_:.4f}")
        cv_scores = grid.cv_results_['mean_test_score']
    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
        model.fit(X_train, y_train)
        st.write(f"Mean CV score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Metrics
    if task=="Classification":
        y_pred = model.predict(X_test)
        st.write("Accuracy:", accuracy_score(y_test,y_pred))
        st.text(classification_report(y_test,y_pred))
    elif task=="Regression":
        y_pred = model.predict(X_test)
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test,y_pred)))
        st.write("R²:", r2_score(y_test,y_pred))
    else:
        st.write("Labels:", getattr(model,"labels_",model.predict(X)))

    # CV Plot
    fig, ax = plt.subplots()
    ax.plot(range(1,len(cv_scores)+1),cv_scores,marker='o')
    ax.set_xlabel("Fold"); ax.set_ylabel("Score"); ax.set_title("CV Scores")
    st.pyplot(fig)

    # Feature importance
    if hasattr(model,"feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots()
        ax.bar(range(len(importances)), importances[indices])
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels([X.columns[i] for i in indices],rotation=45,ha="right")
        ax.set_title("Feature Importances")
        st.pyplot(fig)

    # 2D / 3D visualization
    if X.shape[1]==2:
        x_min,x_max = X.iloc[:,0].min()-1,X.iloc[:,0].max()+1
        y_min,y_max = X.iloc[:,1].min()-1,X.iloc[:,1].max()+1
        xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
        if task=="Classification":
            Z = model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
            plt.contourf(xx,yy,Z,alpha=0.4)
            plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y,edgecolor='k')
            st.pyplot(plt.gcf())
        elif task=="Clustering":
            Z = model.fit_predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
            plt.contourf(xx,yy,Z,alpha=0.4)
            plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y,edgecolor='k')
            st.pyplot(plt.gcf())
    elif X.shape[1]==3:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(X.iloc[:,0],X.iloc[:,1],X.iloc[:,2],c=y)
        st.pyplot(fig)

    # Export model
    with open("model.pkl","wb") as f: pickle.dump(model,f)
    st.success("Model exported as model.pkl")

# Import model
uploaded_model = st.file_uploader("Upload Model (.pkl)", type=["pkl"])
if uploaded_model:
    loaded_model = pickle.load(uploaded_model)
    st.success("Model loaded!")
    if task!="Regression": st.write("Score on test data:", loaded_model.score(X_test,y_test))
