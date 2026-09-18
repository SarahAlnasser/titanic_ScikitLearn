##imports##
# data
import pandas as pd #load in and create a dataframe
import numpy as np #helps when working with machine learning models and deep learning (converts a list of data into an array)

# ML
from sklearn.model_selection import train_test_split, GridSearchCV #allows to create a training and a testing set , allows us to test multiple scenarios
from sklearn.preprocessing import MinMaxScaler #cleaning the data for the model
from sklearn.neighbors import KNeighborsClassifier #the model used
from sklearn.metrics import accuracy_score, confusion_matrix #test model performance

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# dataframe
data = pd.read_csv("titanic.csv") #this represents all our data
data.info() #to know what type of data we are working with
print(data.isnull().sum()) #get how many missing values

##data cleaning and feature engineering##
def preprocess_data(df):
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True) #removes unwanted data

    df["fare"].fillna("0.00", inplace=True) #fill in empty fare slots
    df.drop(columns=["fare"], inplace=True)

    fill_missing_ages(df)

    # convert gender
    df["Sex"] = df["Sex"].map({'male':1,"female":0})

    #feature engineering
    df["FamilySize"] = df["SibSp"] + ["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = np.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)


##fill in missing ages##
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()

    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)

    data = preprocess_data(data)

##create features/target variables(make flashcards)##
x = data.drop(columns=["Survived"])
y = data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

##ML pre-processing##
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

##hyperparemeter tuning##

##predictions and evaluate##

##plot##