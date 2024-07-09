#This file will format and work with the data found in a banking dataset which focuses on marketing targets.
#Clients have 16 features (both categorical and continuous) and the data is used to solve a binary classification
#problem - whether or not the client will subscribe to a term deposit (yes or no)
#Link to Dataset on Kaggle: https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets/data
# Reference: S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
#Note: An additional 'year' column was manually added to the file for further clarity, but not used in the analysis

import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle

df = pd.read_csv(r"C:\Users\VijayBehal\Desktop\Personal Projects\Bank Marketing Target Analysis\archive\train.csv")

#Giving columns more descriptive names, details for each feature found on the Kaggle page
df = df.rename({'default': 'credit in default', 'balance': 'average yearly balance', 'housing': 'housing loan', 
                'contact': 'communication type', 'duration': 'last contact duration in seconds',
                'loan': 'personal loan', 'campaign': 'times contacted during campaign', 
                'pdays': 'days since last contact', 'previous': 'times contacted before campaign', 
                'poutcome': 'previous campaign deposit', 'y': 'deposit'}, axis=1)

#Defining categorical and continuous variables, and output
categorical_columns = ['job', 'marital', 'education', 'credit in default', 'housing loan', 'personal loan', 'communication type', 'month', 'previous campaign deposit']
continuous_columns = ['average yearly balance', 'day', 'last contact duration in seconds', 'times contacted during campaign', 'days since last contact', 'times contacted before campaign']
y_column = ['deposit']

#Converting categorical columns to category data types
for category in categorical_columns:
    df[category] = df[category].astype('category')

#Shuffle dataset
df = shuffle(df)
df.reset_index(drop=True, inplace=True)

#Setting embedding sizes to represent categorical variables in lesser dimensions
categorical_sizes = [len(df[column].cat.categories) for column in categorical_columns]
embedding_sizes = [(size, min(50, (size+1)//2)) for size in categorical_sizes]

#Create arrays of categorical values
job = df['job'].cat.codes.values
marital = df['marital'].cat.codes.values
edu = df['education'].cat.codes.values
cid = df['credit in default'].cat.codes.values
house = df['housing loan'].cat.codes.values
personal = df['personal loan'].cat.codes.values
comm = df['communication type'].cat.codes.values
month = df['month'].cat.codes.values
pcd = df['previous campaign deposit'].cat.codes.values

#Combine into single array and convert to tensor
categorical_data = np.stack([job, marital, edu, cid, house, personal, comm, month, pcd], axis=1)
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

#Create array of continuous values and convert to tensor
continuous_data = np.stack([df[column].values for column in continuous_columns], axis=1)
continuous_data = torch.tensor(continuous_data, dtype=torch.float)

#Create label tensor, flatten so it can be used with Cross Entropy Loss function
#Convert yes/no to 1/0 to allow for creation of tensor of labels
df['deposit'] = df['deposit'].replace({'no': 0, 'yes': 1}).astype(int)
y_column = ['deposit']
y = torch.tensor(df[y_column].values, dtype=torch.long).flatten()

#Create training and test sets. Data previously shuffled
batch = 45211 #size of data set
test = 4521 #10% of data set size
categorical_training = categorical_data[:batch-test]
categorical_test = categorical_data[batch-test:batch]
continuous_training = continuous_data[:batch-test]
continuous_test = continuous_data[batch-test:batch]
y_training = y[:batch-test]
y_test = y[batch-test:batch]
