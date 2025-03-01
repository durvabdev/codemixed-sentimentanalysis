import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, h5py, pickle, re
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torcheval.metrics import MulticlassF1Score
from bpemb import BPEmb

MAXLEN = 200
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

vocabsize = 10000
bpemb_en = BPEmb(lang = "en", dim = 300, vs=vocabsize)
bpemb_bn = BPEmb(lang = "bn", dim = 300, vs=vocabsize)
bpemb_hi = BPEmb(lang = "hi", dim = 300, vs=vocabsize)
totalvocabsize = bpemb_en.vocab_size + bpemb_bn.vocab_size + bpemb_hi.vocab_size
print("here")
embedding_matrix = np.zeros((totalvocabsize, embedding_size))
right = bpemb_en.vocab_size
left = 0
embedding_matrix[left:right] = bpemb_en.vectors
left = right
right = right + bpemb_bn.vocab_size
embedding_matrix[left:right] = bpemb_bn.vectors
left = right
right = right + bpemb_hi.vocab_size
embedding_matrix[left:right] = bpemb_hi.vectors
embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
print(embedding_matrix[:10])

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

class CharLSTM(nn.Module):
    def __init__(self, max_features, maxlen, embedding_size, filter_length, nb_filter, pool_length, lstm_output_size, numclasses):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.conv1d = nn.Conv1d(embedding_size, nb_filter, kernel_size=filter_length, padding='valid')
        self.maxpool1d = nn.MaxPool1d(kernel_size=pool_length)
        self.lstm1 = nn.LSTM(nb_filter, lstm_output_size, batch_first=True, dropout=0.2)
        self.lstm2 = nn.LSTM(lstm_output_size, lstm_output_size, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(lstm_output_size, numclasses)
        #self.softmax = nn.Softmax(dim=1)
        self.numclasses = numclasses

    def forward(self, x):
        x = self.embedding(x.long())
        x = x.permute(0, 2, 1)  # Change dimension order for Conv1d
        x = self.conv1d(x)
        x = self.maxpool1d(x)
        x = x.permute(0, 2, 1)  # Change dimension order back for LSTM
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last output of the sequence
        x = self.fc(x)
        #x = self.softmax(x)
        return x

def train_model(model, X_train, y_train, args):
    max_features, maxlen, embedding_size, filter_length, nb_filter, pool_length, lstm_output_size, batch_size, nb_epoch, numclasses, test_size = args
    early_stopper = EarlyStopper(patience=5, min_delta=0.2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=1e-6)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, stratify= y_train, test_size=test_size, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(nb_epoch):
        total_train_loss = 0.0
        total_batches = 0
        correct_train_preds = 0
        total_train_samples = 0
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            #for name, param in model.named_parameters():
            #    if param.grad is None:
            #        print(f"No gradient for {name}")
            optimizer.step()
            total_train_loss += loss.item()
            total_batches += 1
            predictions = outputs.argmax(dim=1)
            correct_train_preds += (predictions == batch_y).sum().item()
            total_train_samples += batch_y.size(0)

        avg_train_loss = total_train_loss / total_batches
        train_acc = correct_train_preds / total_train_samples

        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.tensor(X_valid).to(device))
            val_loss = criterion(val_outputs, torch.tensor(y_valid).to(device))
            val_acc = (val_outputs.argmax(dim=1) == torch.tensor(y_valid).to(device)).float().mean()

           # Calculate F1 score
            f1_metric = MulticlassF1Score(num_classes=model.numclasses, average=None) # Initialize the metric object with averaging method
            f1_metric.update(val_outputs.argmax(dim=1), torch.tensor(y_valid).to(device))  # Update the metric with predictions and targets
            f1_score = f1_metric.compute() # Compute the F1 score

        #scheduler.step(val_loss)
        if early_stopper.early_stop(val_loss):
            break

        print(f'Epoch {epoch+1}/{nb_epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Train Acc: {train_acc:.4f}')
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

def wrapper(X_train, y_train):
    print(X_train.shape, y_train.shape)
    print('Creating LSTM Network...')
    model = CharLSTM(MAX_FEATURES, MAXLEN, embedding_size, filter_length, nb_filter, pool_length, lstm_output_size, numclasses)
    print(y_train.shape)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = test_size, stratify=y_train, random_state=42)

    model = train_model(model, X_train, y_train,   [MAX_FEATURES, MAXLEN, embedding_size,
                 filter_length, nb_filter, pool_length, lstm_output_size, batch_size,
                 number_of_epochs, numclasses, test_size])

    print('Evaluating model...')
    print(embedding_matrix[:10])
    evaluate_model(X_test, y_test, model, batch_size, numclasses)