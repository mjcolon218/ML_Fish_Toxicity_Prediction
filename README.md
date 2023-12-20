![screenshots1](images/fishimage3.jpg?raw=true)

# ML_Fish_Toxicity_Prediction
Welcome to ML_Fish_Toxicity_Predictor, an open-source data analysis and machine learning project! This repository hosts a comprehensive analysis of a real-world dataset, showcasing the entire data science pipeline from data exploration to model building.


## Modules/Libraries
* Scikit-learn
* MatPlotLib
* Statistics
* Numpy
* Pandas
* Seaborn
* Scipy

## # Dataset Overview
#### The dataset comprises several molecular descriptors used to predict Fish Toxicity (LC50). The features include:
##### * MLOGP (Molar Log P): This is a measure of the compound's lipophilicity, often represented as the logarithm of the octanol/water partition coefficient (log P). Lipophilicity is a key factor in bioaccumulation and permeability through biological membranes, which can directly influence a compound's toxicity.

##### * CIC0 (Information Indices): These are topological indices derived from the molecular graph of the compound. They provide information about the compound's molecular structure without explicit reference to 3D shape. Such indices often correlate with various physicochemical properties and biological activities.

##### * GATS1i (2D Autocorrelations): Geary Autocorrelation of lag 1 weighted by ionization potential. It's a 2D autocorrelation descriptor that captures information about molecular topology and electron distribution. This descriptor can be related to how a molecule interacts with biological targets.

##### * NdssC (Atom-Type Counts - Sulfur): This descriptor counts the number of sulfur atoms in the molecule. The presence of certain types of atoms (like sulfur) can influence the reactivity and hence the toxicity of a molecule.

##### * NdsCH (Atom-Type Counts - CH Groups): This refers to the count of specific types of carbon-hydrogen groups in the molecule. The structure and count of such groups can affect a molecule's physical and chemical properties, influencing its biological activity.

##### * SM1_Dz(Z) (2D Matrix-Based Descriptors): This is a descriptor derived from a 2D matrix representation of the molecular structure. It typically captures information about the spatial arrangement of atoms or functional groups within the molecule, influencing its chemical behavior and interactions in a biological context.



## Univariate Analysis / Skewness / Distributions

![screenshots1](images/skewimage.png?raw=true)

### CIC0         0.045458
### M1_Dz(Z)    0.695090
### GATS1i       0.723107
### NdssC        3.400815
### NdsCH        2.239090
### MLOGP       -0.035191

## Skewness Analysis
#### The skewness values and the corresponding histograms for each feature (excluding the target variable) are as follows:

#### MLOGP:

#### Skewness: -0.03
#### The distribution appears fairly symmetrical, as indicated by the low skewness value.
#### CIC0:

#### Skewness: 0.04
#### This shows a moderate right skew. A transformation might help in normalizing this distribution.
#### GATS1i:

#### Skewness: 0.72
#### Similar to CIC0, this feature also has a moderate right skew.
#### NdssC:

#### Skewness: 3.40
#### This feature shows a high positive skewness, indicating a strong right skew. A transformation is likely needed.
#### NdsCH:

#### Skewness: 2.24
#### Another highly positively skewed distribution, suggesting the need for transformation.
#### SM1_Dz(Z):

#### Skewness: 0.70
#### The distribution is almost symmetrical.

### Boxplot to check for outliers.

![screenshots1](images/boxplots.png?raw=true)
#### Box Plots: The box plots for "NdssC" and "NdsCH" against the target variable (fish toxicity) provide insights into the central tendency and spread of the target variable across different categories. They also highlight outliers in each category.

#### Violin Plots: The violin plots give a more detailed view of the target variable's distribution across different categories of "NdssC" and "NdsCH". The width of the violin plot at different values indicates the density of data points, offering a more nuanced understanding of the distribution.

#### These visualizations are particularly useful for discrete variables as they reveal patterns and distributions that might not be evident in scatter plots. They can also guide feature engineering and selection in machine learning models, indicating which categories of these discrete variables have more influence on the target variable.

## Bi-Variate Analysis
#### Strip plots show you where the data is dense at.
![screenshots1](images/striplot.png?raw=true)
![screenshots1](images/pairplot.png?raw=true)
![screenshots1](images/facetgrid.png?raw=true)


## * Scatter plots of each feature against Fish Toxicity show diverse patterns, suggesting different levels of linear or nonlinear relationships.
## The correlation matrix indicates the degree of linear relationship between features and with Fish Toxicity. For instance, some features might have a moderate positive or negative correlation with Fish Toxicity, suggesting their potential influence in predicting toxicity.
![screenshots1](images/correlation.png?raw=true)
## Facet Grid (MLOGP vs Fish Toxicity by NdssC):

#### A series of scatter plots, each representing a different value of "NdssC", showing the relationship between "MLOGP" and Fish Toxicity.
##### Finding: This grid allows for the comparison of relationships across different "NdssC" categories. It may reveal whether the relationship between "MLOGP" and Fish Toxicity changes with different atom-type counts.

![screenshots1](images/facetgrid.png?raw=true)

### Facet Grids and Joint Plots
### Facet grids and joint plots allowed for a detailed examination of relationships between specific pairs of features and Fish Toxicity, across different categories of other features (like "NdssC").
![screenshots1](images/jointplot.png?raw=true)

## 3D Plots
## 3D visualizations provided an enhanced view of the interactions among multiple molecular descriptors, offering insights into the complex relationships that might influence Fish Toxicity.

![screenshots1](images/3d.png?raw=true)
![screenshots1](images/3d2.png?raw=true)
![screenshots1](images/3d3.png?raw=true)
![screenshots1](images/3d4.png?raw=true)
![screenshots1](images/3d5.png?raw=true)

## Machine Learning Modeling 










## Interpretation:
### The Random Forest Regression model shows the best performance among the three, with the lowest MSE and RMSE, and the highest R-squared value. This indicates it's better at capturing the complexity and non-linear relationships in the data.
```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/fish_transformed.csv')
# Separating features and target
X = data.drop('Fish Toxicity', axis=1)
y = data['Fish Toxicity']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the models
linear_model = LinearRegression()
ridge_model = Ridge(alpha=1.0)
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Function to train and evaluate a model
def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Training the model
    model.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2

# Training and evaluating Linear Regression
linear_mse, linear_rmse, linear_r2 = train_evaluate_model(linear_model, X_train, y_train, X_test, y_test)

# Training and evaluating Ridge Regression
ridge_mse, ridge_rmse, ridge_r2 = train_evaluate_model(ridge_model, X_train, y_train, X_test, y_test)

# Training and evaluating Random Forest Regression
rf_mse, rf_rmse, rf_r2 = train_evaluate_model(random_forest_model, X_train, y_train, X_test, y_test)

# Print the results
print("Linear Regression: MSE =", linear_mse, ", RMSE =", linear_rmse, ", R^2 =", linear_r2)
print("Ridge Regression: MSE =", ridge_mse, ", RMSE =", ridge_rmse, ", R^2 =", ridge_r2)
print("Random Forest Regression: MSE =", rf_mse, ", RMSE =", rf_rmse, ", R^2 =", rf_r2)

Linear Regression: MSE = 1.0463821653353844 , RMSE = 1.0229282307842444 , R^2 = 0.5675170258323361
Ridge Regression: MSE = 1.0463854210971713 , RMSE = 1.0229298221760725 , R^2 = 0.56751568018484
Random Forest Regression: MSE = 1.0331873061963945 , RMSE = 1.0164582166505391 , R^2 = 0.572970627884436

```

### Both the Linear Regression and Ridge Regression models have similar performance metrics, with relatively higher MSE and RMSE, and lower R-squared values compared to the Random Forest model. This could suggest that the relationship between the predictors and the target variable has non-linear aspects that these linear models are not fully capturing.
![screenshots1](images/Lregression.png?raw=true)
![screenshots1](images/modelcoeffecientvisuals.png?raw=true)
### The R-squared values for all models are moderate, indicating that while the models have some predictive power, there's still room for improvement. Feature engineering, model tuning, or trying different modeling approaches might help to increase the predictive performance.