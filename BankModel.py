#This file defines the model used to predict whether or not a client will subscribe to a term deposit

import torch
import torch.nn as nn

class TabularBankModel(nn.Module):

    #embedding_sizes: list of tuples, each categorical variable size paired with an embedding size
    #num_continuous: number of continuous variables
    #output_size: output size
    #layers: layer sizes in a list
    #p: dropout probability for each dropout layer. Default=0.5
    
    def __init__(self, embedding_sizes, num_continuous, output_size, layers, p=0.5):
        super().__init__()

        #set up embedded layers. Categorical data will be filtered through these
        #embeddings in the forward method
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in embedding_sizes])

        #Set up dropout layer for embeddings
        self.emb_drop = nn.Dropout(p)

        #Set up batch normalization for continuous variables to fall within same magnitude range
        self.bn_cont = nn.BatchNorm1d(num_continuous)

        #Setting up sequence of neural network layers where each level includes a linear
        #function, activation function (ReLU), normalization step, and a dropout layer.
        #We will combine the list of layers

        #store layers
        layerlist = []

        #number of total embeddings
        num_emb = sum([nf for ni,nf in embedding_sizes])

        #number of inputs = number of embeddings (categorical) + number of continuous
        n_in = num_emb + num_continuous

        #set up layers
        #layers is list of layer sizes, 'i' will keep correct output for each layer
        # as input of next layer as we change 'i' at the end
        # Example layers variable can look like: [200, 100, 150]
        for i in layers:
            #'i' starts as first output
            layerlist.append(nn.Linear(n_in, i))
            #ReLU Activation Function
            layerlist.append(nn.ReLU(inplace=True))
            #Batch normalization on input to next layer
            layerlist.append(nn.BatchNorm1d(i))
            #Dropout with some probability
            layerlist.append(nn.Dropout(p))
            #Set next input to current output
            n_in = i

        #Last layer
        layerlist.append(nn.Linear(layers[-1], output_size))

        #Combine all layers
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):

        embeddings = []

        #pass each column of features into respective embedding and append to list
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))

        #concatenate embeddings into one tensor
        x = torch.cat(embeddings, 1)
        #Dropout for embeddings
        x = self.emb_drop(x)

        #Batch normalization for continuous variables
        x_cont = self.bn_cont(x_cont)

        #Concatenate categorical and continuous data
        x = torch.cat([x,x_cont],1)

        #Forward
        x = self.layers(x)
        return x