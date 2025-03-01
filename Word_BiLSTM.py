import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from torcheval.metrics import MulticlassF1Score

################# GLOBAL VARIABLES #####################
# Data I/O formatting
# MAXLEN = 200
embedding_size = 768

# Model Parameters
batch_size = 128
number_of_epochs = 50
numclasses = 3
test_size = 0.2


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class BiLSTMModel(nn.Module):
    def __init__(self, hidden_size, numclasses):
        super(BiLSTMModel, self).__init__()
        # Since we are treating each element of the 768-vector as a time step,
        # input_size for LSTM becomes 1 and sequence length becomes 768.
        self.bilstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                              bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        # The LSTM has a bidirectional output, so its output size is hidden_size * 2.
        self.dense1 = nn.Linear(hidden_size * 2, 100)
        self.dense2 = nn.Linear(100, numclasses)
        self.relu = nn.ReLU()
        # We will apply softmax along the classes dimension (dim=2)
        self.softmax = nn.Softmax(dim=2)
        self.numclasses = numclasses

    def forward(self, x):
        # x has shape: [batch_size, 768]
        # Treat each element of the vector as a time step by reshaping to [batch_size, 768, 1]
        x = x.unsqueeze(-1)  # new shape: [batch_size, 768, 1]

        # Process the sequence with a bidirectional LSTM.
        lstm_out, _ = self.bilstm(x)  # output shape: [batch_size, 768, hidden_size * 2]
        lstm_out = self.dropout(lstm_out)

        # Apply the first dense layer with ReLU activation.
        dense1_out = self.relu(self.dense1(lstm_out))  # shape: [batch_size, 768, 100]

        # Apply the second dense layer to get logits for each time step.
        dense2_out = self.dense2(dense1_out)  # shape: [batch_size, 768, numclasses]

        # Convert logits into a probability distribution for each time step.
        output = self.softmax(dense2_out)  # shape remains: [batch_size, 768, numclasses]

        # Finally, aggregate over the time dimension (e.g., by taking the mean)
        # to obtain one output vector per sentence.
        output = output.mean(dim=1)  # final shape: [batch_size, numclasses]
        return output


def train_model(model, X_train, y_train, args):
    embedding_size, batch_size, nb_epoch, numclasses, test_size = args
    early_stopper = EarlyStopper(patience=5, min_delta=0.2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify=y_train, test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
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
            val_outputs = model(torch.tensor(X_valid, dtype=torch.float32).to(device))
            val_loss = criterion(val_outputs, torch.tensor(y_valid).to(device))
            val_acc = (val_outputs.argmax(dim=1) == torch.tensor(y_valid).to(device)).float().mean()

            f1_metric = MulticlassF1Score(num_classes=model.numclasses, average=None)
            f1_metric.update(val_outputs.argmax(dim=1), torch.tensor(y_valid).to(device))
            f1_score = f1_metric.compute()

        print(f'Epoch {epoch + 1}/{nb_epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print('F-1 score is ', f1_score)

    return model


def evaluate_model(X_test, y_test, model, batch_size, numclasses):
    device = next(model.parameters()).device
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test).to(device)

    with torch.no_grad():
        outputs = model(X_test)
        predicted = outputs.argmax(dim=1)
        accuracy = (predicted == y_test).float().mean()

        f1_metric = MulticlassF1Score(num_classes=model.numclasses, average=None)
        f1_metric.update(predicted, y_test)
        f1_score = f1_metric.compute()

    print(f'Test accuracy: {accuracy:.4f}')
    print(f'F1 score: {f1_score}')


def wrapper(X_train, y_train):
    print('Creating Bi-LSTM Network...')
    model = BiLSTMModel(hidden_size=300, numclasses=numclasses)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=test_size, random_state=42)

    # Assuming X_train and y_train are your training data and labels
    model = train_model(model, X_train, y_train, [embedding_size, batch_size,
                                                  number_of_epochs, numclasses, test_size])

    print('Evaluating model...')
    # Assuming X_test and y_test are your test data and labels
    evaluate_model(X_test, y_test, model, batch_size, numclasses)
