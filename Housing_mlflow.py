import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32

import logging
import tarfile
import urllib.request
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import pandas as pd
import csv

"""## DATA PREPROCESSING"""

housing=pd.read_csv('housing_.csv')

housing.head()


def split_train_test(data, test_ratio):
  shuffled_indices = np.random.permutation(len(data))
  test_set_size = int(len(data) * test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)



def test_set_check(identifier, test_ratio):
  return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
def split_train_test_by_id(data, test_ratio, id_column):
  ids = data[id_column]
  in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
  return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")



housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")



""" **Using Scikit-Learn**"""

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

"""Suppose if median income is important in predicting median housing prices."""

housing["income_cat"] = pd.cut(housing["median_income"],
bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
 set_.drop("income_cat", axis=1, inplace=True)

"""## Visualize the Data for insights"""

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude")

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,6),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()

"""This tells that house prices are more in the locations close to the ocean and to population density."""

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
alpha=0.1)

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]



corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

"""Preparing the Data for Machine Learning Algos"""

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()



sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

imputer.statistics_

X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)

"""## Handling Categorical Attributes
"""

housing_cat = housing[["ocean_proximity"]]



from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

housing_cat_1hot.toarray()

"""## Custom Tranformers"""

from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)



"""## Feature Scaling"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)

housing_num_tr

housing_prepared

"""For reference, here is the old solution based on a `DataFrameSelector` transformer (to just select a subset of the Pandas `DataFrame` columns), and a `FeatureUnion`:"""

from sklearn.base import BaseEstimator, TransformerMixin


# Create a class to select numerical or categorical columns
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


"""Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:"""

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline(
    [
        ("selector", OldDataFrameSelector(num_attribs)),
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

old_cat_pipeline = Pipeline(
    [
        ("selector", OldDataFrameSelector(cat_attribs)),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ]
)

from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(
    transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ]
)

old_housing_prepared = old_full_pipeline.fit_transform(housing)
old_housing_prepared

"""## Select and Train a Model"""

with mlflow.start_run(run_name="Parent Run"):
    with mlflow.start_run(run_name="Linear Regression", nested=True):
        from sklearn.linear_model import LinearRegression

        print("\nLinear Regression")

        lin_reg = LinearRegression()
        lin_reg.fit(housing_prepared, housing_labels)

        housing_predictions = lin_reg.predict(housing_prepared)

        from sklearn.metrics import mean_squared_error

        lin_mse = mean_squared_error(housing_labels, housing_predictions)
        lin_rmse = np.sqrt(lin_mse)

        from sklearn.metrics import mean_absolute_error

        lin_mae = mean_absolute_error(housing_labels, housing_predictions)

        from sklearn.metrics import r2_score

        r2 = r2_score(housing_labels, housing_predictions)

        print("Root Mean Squared Error:", lin_rmse)
        print("Mean Absolute Error:", lin_mae)
        print("R2: %s" % r2)

        from sklearn.model_selection import cross_val_score

        lin_scores = cross_val_score(
            lin_reg,
            housing_prepared,
            housing_labels,
            scoring="neg_mean_squared_error",
            cv=10,
        )
        lin_rmse_scores = np.sqrt(-lin_scores)

        def display_scores(scores):
            print("Scores:", scores)
            print("Mean:", scores.mean())
            print("Standard deviation:", scores.std())

        display_scores(lin_rmse_scores)

        mlflow.log_param("Scores", lin_rmse_scores)
        mlflow.log_metric("Mean", lin_rmse_scores.mean())
        mlflow.log_metric("RMSE", lin_rmse)
        mlflow.log_metric("MAE", lin_mae)
        mlflow.log_metric("Standard deviation", lin_rmse_scores.std())
        mlflow.log_metric("R2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lin_reg, "model", registered_model_name="LinearRegressorModel"
            )
        else:
            mlflow.sklearn.log_model(lin_reg, "model")

    with mlflow.start_run(run_name="Decision Tree Regressor", nested=True):
        from sklearn.tree import DecisionTreeRegressor

        print("\nDecision Tree Regressor")

        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(housing_prepared, housing_labels)

        housing_predictions = tree_reg.predict(housing_prepared)

        tree_mse = mean_squared_error(housing_labels, housing_predictions)
        tree_rmse = np.sqrt(tree_mse)

        tree_mae = mean_absolute_error(housing_labels, housing_predictions)

        r2 = r2_score(housing_labels, housing_predictions)

        print("Mean Squared Error:", tree_rmse)
        print("Mean Absolute Error:", tree_mae)
        print("R2: %s" % r2)

        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            tree_reg,
            housing_prepared,
            housing_labels,
            scoring="neg_mean_squared_error",
            cv=10,
        )
        tree_rmse_scores = np.sqrt(-scores)

        def display_scores(scores):
            print("Scores:", scores)
            print("Mean:", scores.mean())
            print("Standard deviation:", scores.std())

        display_scores(tree_rmse_scores)

        mlflow.log_param("Scores", tree_rmse_scores)
        mlflow.log_metric("Mean", tree_rmse_scores.mean())
        mlflow.log_metric("RMSE", tree_rmse)
        mlflow.log_metric("MAE", tree_mae)
        mlflow.log_metric("Standard deviation", tree_rmse_scores.std())
        mlflow.log_metric("R2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                tree_reg, "model", registered_model_name="DecisionTreeRegressorModel"
            )
        else:
            mlflow.sklearn.log_model(tree_reg, "model")

    with mlflow.start_run(run_name="Random Forest Regressor", nested=True):
        from sklearn.ensemble import RandomForestRegressor

        print("\nRandom Forest Regressor")

        forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        forest_reg.fit(housing_prepared, housing_labels)

        housing_predictions = forest_reg.predict(housing_prepared)

        forest_mse = mean_squared_error(housing_labels, housing_predictions)
        forest_rmse = np.sqrt(forest_mse)

        forest_mae = mean_absolute_error(housing_labels, housing_predictions)

        r2 = r2_score(housing_labels, housing_predictions)

        print("Root Mean Squared Error:", forest_rmse)
        print("Mean Absolute Error:", forest_mae)
        print("R2: %s" % r2)

        from sklearn.model_selection import cross_val_score

        forest_scores = cross_val_score(
            forest_reg,
            housing_prepared,
            housing_labels,
            scoring="neg_mean_squared_error",
            cv=10,
        )
        forest_rmse_scores = np.sqrt(-forest_scores)

        def display_scores(scores):
            print("Scores:", scores)
            print("Mean:", scores.mean())
            print("Standard deviation:", scores.std())

        display_scores(forest_rmse_scores)

        mlflow.log_param("Scores", forest_rmse_scores)
        mlflow.log_metric("Mean", forest_rmse_scores.mean())
        mlflow.log_metric("RMSE", forest_rmse)
        mlflow.log_metric("MAE", forest_mae)
        mlflow.log_metric("Standard deviation", forest_rmse_scores.std())
        mlflow.log_metric("R2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(
                forest_reg, "model", registered_model_name="RandomForestRegressorModel"
            )
        else:
            mlflow.sklearn.log_model(forest_reg, "model")

    with mlflow.start_run(run_name="Support Vector Machines", nested=True):
        from sklearn.svm import SVR

        print("\nSVR")
        svm_reg = SVR(kernel="linear")
        svm_reg.fit(housing_prepared, housing_labels)

        housing_predictions = svm_reg.predict(housing_prepared)

        svm_mse = mean_squared_error(housing_labels, housing_predictions)
        svm_rmse = np.sqrt(svm_mse)

        svm_mae = mean_absolute_error(housing_labels, housing_predictions)

        r2 = r2_score(housing_labels, housing_predictions)

        print("Root Mean Squared Error:", svm_rmse)
        print("Mean Absolute Error:", svm_mae)
        print("R2: %s" % r2)

        from sklearn.model_selection import cross_val_score

        svm_scores = cross_val_score(
            svm_reg,
            housing_prepared,
            housing_labels,
            scoring="neg_mean_squared_error",
            cv=10,
        )
        svm_rmse_scores = np.sqrt(-svm_scores)

        def display_scores(scores):
            print("Scores:", scores)
            print("Mean:", scores.mean())
            print("Standard deviation:", scores.std())

        display_scores(svm_rmse_scores)

        mlflow.log_param("Scores", svm_rmse_scores)
        mlflow.log_metric("Mean", svm_rmse_scores.mean())
        mlflow.log_metric("RMSE", svm_rmse)
        mlflow.log_metric("MAE", svm_mae)
        mlflow.log_metric("Standard deviation", svm_rmse_scores.std())
        mlflow.log_metric("R2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(svm_reg, "model", registered_model_name="SVMModel")
        else:
            mlflow.sklearn.log_model(svm_reg, "model")
