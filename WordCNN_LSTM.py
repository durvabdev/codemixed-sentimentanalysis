## Token level: word
## Pre-trained embeddings for each word
## no aggregation of word embeddings. each sentence is a list of word vectors (ie., 2D matrix)

##DRI ARCH

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import pickle
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import re
from torcheval.metrics import MulticlassF1Score

################# GLOBAL VARIABLES #####################
# Filenames
Masterdir = '/content/'  # Colab working directory
Datadir = 'sample_data/'
Modeldir = 'Models/'
Featuredir = 'Features/'
inputdatasetfilename = 'IIITH_Codemixed.txt'
exp_details = 'new_experiment'

# Data I/O formatting
SEPERATOR = '\t'
DATA_COLUMN = 1
LABEL_COLUMN = 3
LABELS = ['0','1','2'] # 0 -> Negative, 1-> Neutral, 2-> Positive
mapping_char2num = {}
mapping_num2char = {}
MAXLEN = 200

# LSTM Model Parameters
MAX_FEATURES = 0
embedding_size = 300
filter_length = 3
nb_filter = 128
pool_length = 3
lstm_output_size = 128
batch_size = 128
number_of_epochs = 50
numclasses = 3
test_size = 0.2

class CharLSTM(nn.Module):
    def __init__(self, max_features, maxlen, embedding_size, filter_length, nb_filter, pool_length, lstm_output_size, numclasses):
        super(CharLSTM, self).__init__()
        ##embedding dimension is considered as channels
        ## kernel size indicates how many words are convoluted together
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=nb_filter, kernel_size=filter_length, padding='valid')
        ##the convolution gives output diension (batch_size, nb_filter, new sequence length)
        ## if nb_filter = 1, conv1D automatically gives a 1D vector per sentence
        
        ## maxpool1D takes input of shape (batch_size, nb_filter, new sequence length)
        ## and it slides over new sequence length, giving maximum value for each channel. it basically reduces the third dimension (new seqquence length)
        self.maxpool1d = nn.MaxPool1d(kernel_size=pool_length)
        self.lstm1 = nn.LSTM(nb_filter, lstm_output_size, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(lstm_output_size, lstm_output_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(lstm_output_size, numclasses)
        self.softmax = nn.Softmax(dim=1)
        self.numclasses = numclasses

    def forward(self, x):
        x = x.unsqueeze(1)
        #x = x.permute(0, 2, 1)  # Change dimension order for Conv1d
        x = self.conv1d(x)
        x = self.maxpool1d(x)
        x = x.permute(0, 2, 1)  # Change dimension order back for LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last output of the sequence
        x = self.fc(x)
        x = self.softmax(x)
        return x


def train_model(model, X_train, y_train, args):
    max_features, maxlen, embedding_size, filter_length, nb_filter, pool_length, lstm_output_size, batch_size, nb_epoch, numclasses, test_size = args

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(nb_epoch):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_valid).to(device))
            val_loss = criterion(val_outputs, torch.tensor(y_valid).to(device))
            val_acc = (val_outputs.argmax(dim=1) == torch.tensor(y_valid).to(device)).float().mean()

           # Calculate F1 score
            f1_metric = MulticlassF1Score(num_classes=model.numclasses, average=None) # Initialize the metric object with averaging method
            f1_metric.update(val_outputs.argmax(dim=1), torch.tensor(y_valid).to(device))  # Update the metric with predictions and targets
            f1_score = f1_metric.compute() # Compute the F1 score

        print(f'Epoch {epoch+1}/{nb_epoch}, Val Loss: {val_loss:.4f}, Val Accs: {val_acc:.4f}')
        print('F-1 score is ', f1_score, "Average", f1_score.mean())

    return model

def evaluate_model(X_test, y_test, model, batch_size, numclasses):
    device = next(model.parameters()).device
    model.eval()
    X_test = torch.tensor(X_test).to(device)
    y_test = torch.tensor(y_test).to(device)

    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean()

        # Calculate F1 score
        f1_metric = MulticlassF1Score(num_classes=model.numclasses, average=None) # Initialize the metric object with averaging method
        f1_metric.update(predicted, y_test)  # Update the metric with predictions and targets
        f1_score = f1_metric.compute() # Compute the F1 score

    print(f'Test accuracy: {accuracy:.4f}')
    print(f'F1 score: {f1_score}', "Average is", f1_score.mean())

#if __name__ == '__main__':
def wrapper(X_train, y_train):
    print(X_train.shape, y_train.shape)
    print('Creating LSTM Network...')
    model = CharLSTM(MAX_FEATURES, MAXLEN, embedding_size, filter_length, nb_filter, pool_length, lstm_output_size, numclasses)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=test_size, random_state=42)

    model = train_model(model, X_train, y_train,   [MAX_FEATURES, MAXLEN, embedding_size,
                 filter_length, nb_filter, pool_length, lstm_output_size, batch_size,
                 number_of_epochs, numclasses, test_size])

    print('Evaluating model...')
    evaluate_model(X_test, y_test, model, batch_size, numclasses)