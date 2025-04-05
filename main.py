import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("/Users/sujalkarki/Desktop/Stock_Market_Analysis/Stock.csv")
print(df.head(5))  #check first 5 row of the databases
print(df.isnull().sum()) # check missing value available or not 
print(df.fillna(0)) # fill the missing value by 0 
print(df.shape) #check the shape of the dataset
print(df.info()) # check the information of the data set 
print(df.describe()) #
