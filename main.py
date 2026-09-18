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

##fill in missing ages##

#
#