#This file trains the Tabular Bank Model on the training data and tests it using the testing data

from DataAnalysis import embedding_sizes, categorical_data, continuous_data, categorical_training, continuous_training, categorical_test, continuous_test, y_training, y_test
from BankModel import TabularBankModel
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_SIZE = 2 #binary classification problem
LAYERS = [200,100] #layer sizes, can experiment with this

#Create instance of model
model = TabularBankModel(embedding_sizes, continuous_data.shape[1],OUTPUT_SIZE,LAYERS,p=0.4)

#Define loss function (Cross Entropy) and optimizer (Adam, LearningRate=0.001)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Train the model:

start_time = time.time()

epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_predicted = model(categorical_training, continuous_training)
    loss = criterion(y_predicted, y_training)
    losses.append(loss.detach().numpy())

    #print loss for every 25th epoch
    if i%25 == 1:
        print(f'epoch: {i:3}   loss:  {loss.item():10.8f}')

    optimizer.zero_grad() #reset gradient
    loss.backward() #backpropogation
    optimizer.step() #gradient descent

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

#Plot CE Loss against epochs
plt.plot(range(epochs), losses)
plt.ylabel('Cross Entropy Loss')
plt.xlabel('epoch')

#Evaluate test set
with torch.no_grad():

    y_value = model(categorical_test, continuous_test)

    loss = criterion(y_value, y_test)

print(f'Cross Entropy Loss for Test Data: {loss:.8f}')

#Print percent accuracy for test set
rows = 4521
correct = 0

for i in range(rows):
    if y_value[i].argmax().item() == y_test[i]:
        correct += 1

print(f'\n{correct} out of {rows} = {100*correct/rows:.2f}% correct')

#Save model
if len(losses) == epochs:
    torch.save(model.state_dict(), 'TabularBankModel.pt')
else:
    print("Failed to save model")

plt.show()
