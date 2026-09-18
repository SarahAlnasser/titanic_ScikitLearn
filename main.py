##imports##
# data
import pandas as pd #load in and create a dataframe
import numpy as np #helps when working with machine learning models and deep learning

# ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# dataframe
data = pd.read_csv("titanic.csv") #this represents all our data
data.info() #to know what type of data we are working with
print(data.isnull().sum()) #get how many missing values


##fill in missing ages##
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()

    df["Age"] = df.apply(
        lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"],
        axis=1
    )
    return df


##data cleaning and feature engineering##
def preprocess_data(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"]) #removes unwanted data

    df = fill_missing_ages(df)

    # fill remaining missing values (KNN cannot handle NaN)
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # convert text columns to numbers
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates='drop')
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)

    return df


data = preprocess_data(data)

##create features/target variables##
X = data.drop(columns=["Survived"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

##ML pre-processing##
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##hyperparameter tuning - KNN##
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)
print(f"Best parameters: {best_model.get_params()}")

##predictions and evaluate##
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f"Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

##plot##
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived","Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("predicted value")
    plt.ylabel("true values")
    plt.show(block=True)


plot_model(matrix)