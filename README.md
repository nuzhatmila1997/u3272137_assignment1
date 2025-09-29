# Project overview

End-to-end analysis of Sydney restaurant data:

- EDA and visualisation (static + interactive)
- Geospatial cuisine density (GeoPandas/Plotly)
- Feature engineering for ML
- Regression (Linear Regression + Gradient Descent)
- Classification (LogReg, RF, GBoost, SVM) with full metrics
- PySpark ML replicas for one regression and one classification
- Model artefacts saved to models//

## Installation

### Conda (recommended for GeoPandas/GDAL stack)

```bash
# create + activate env (Python 3.10+)
conda create -n sydfood python=3.10 -y
conda activate sydfood

# core scientific stack
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn -y

# geospatial stack
conda install -c conda-forge geopandas shapely pyproj fiona -y

# interactive viz + spark + utils
conda install -c conda-forge plotly -y
pip install pyspark joblib
```

### Pip (works on most systems with manylinux wheels)

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install --upgrade pip
pip install numpy pandas scikit-learn matplotlib seaborn plotly joblib pyspark
pip install geopandas

```

### Pyspark

- Option A (local): `pip install pyspark`

<!-- Block:

```bash
pip install pyspark
``` -->

If Java errors occur, install OpenJDK (https://learn.microsoft.com/en-us/java/openjdk/install) and retry.

- Option B (Google Colab): `!pip install pyspark` in the first cell.
- Option C (Databricks Community): Create a cluster (Runtime 13+), attach a notebook.

## How to run

You can work from a notebook on vscode or jupyter. The code blocks you built in this project are already segmented by task; run them in order.

### EDA + Feature Engineering [eda-zomato.ipynb]

- Run the cells that:

load Dataset/zomato_df_final_data.csv

inspect dtypes, missingness, and summary stats

build distributions and relationship plots

construct geospatial joins with Dataset/sydney.geojson

- perform feature engineering:

ratings/votes imputation (is_unrated, votes=0, rating_number=0 sentinel, rating_text="Unrated")

price imputation (median by type), drop cost_2

location validation via polygon containment + subzone centroid fill

categorical encodings (cuisine multi-hot top-K+Other, type multi-hot, subzone frequency/target encoding)

### Regression and Classification (Scikit-Learn, PySpark) [predictive-modelling.ipynb]

Run the regression cells to:

- split 80/20

- train LinearRegression

- train Gradient Descent linear regression (standardised)

- print MSE for both

#### Classification (Scikit-Learn)

- Map rating_text → binary:

- Class 0: Poor + Average

- Class 1: Good + Very Good + Excellent

- split 80/20 (stratified)

- train: Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF)

- report confusion matrix, precision, recall, F1 per model

- print a comparison table

#### PySpark ML pipelines

- Build Spark DataFrames from your engineered pandas frames

- Create pipeline:

  - VectorAssembler (+ optional StandardScaler)

  - LinearRegression (regression) / LogisticRegression (classification)

  - Evaluate with MSE (regression) and AUC + P/R/F1 (classification)

#### Save models

- Scikit-Learn (pickles) and Spark pipelines (native)

## What results to expect

### EDA outputs

- Histograms/boxplots for cost & ratings; scatter plots (cost–votes, cost–rating)

- Correlation heatmap

- Cuisine maps:

  - static choropleth or interactive Plotly map (hover suburb counts, zoom/pan)

  - optional static/interactive bubble maps (one point per suburb)

- Feature engineering artifacts

  - Encoded columns like cuisine**\*, type**\*, subzone_freq, log_cost, log_votes, dist_cbd_km, etc.

- Cleaned/imputed coordinates and cost

- A narrowed, numeric feature matrix for modelling

### Regression

Printed MSE for:

- Scikit-Learn LinearRegression

- Gradient Descent Linear Regression

(Typical pattern: GD matches LR when converged; exact numbers depend on your split and features.)

**Classification**

- Confusion matrices and a metrics table (precision, recall, F1) for:

**Logistic Regression**

- Random Forest

- Gradient Boosting

- SVM (RBF)

Expect tree-based models to be competitive; SVM/LogReg performance depends on scaling and class balance.

**PySpark**

- Same metrics (MSE / P-R-F1 / AUC) printed for Spark pipelines

- Speed/scalability: Spark shows benefits with larger-than-memory datasets; on small data it may be slower than scikit-learn due to overhead.

**Model artefacts**

Files in models//:

`sklearn_linear_regression.pkl`, `linear_regression_gd.pkl`

`cls_logreg.pkl`, `cls_random_forest.pkl`, `cls_gradient_boosting.pkl`, `cls_svm_rbf.pkl`

`spark_regression_pipeline/`, `spark_classification_pipeline/`
