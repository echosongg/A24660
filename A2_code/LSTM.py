import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    all_specificities = []

    for i in range(3):  # Loop through classes
        # True negatives are all the samples that are not in the current class (both predicted and actual)
        tn = cm[~i, ~i].sum()

        # False positives are all the samples of other classes that are predicted as the current class
        fp = cm[~i, i].sum()

        # Compute specificity for the current class
        spec = tn / (tn + fp)
        all_specificities.append(spec)

    # Return the mean specificity over all classes
    return np.mean(all_specificities)

# caculate geometric_mean
def geometric_mean(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='macro')
    spec = specificity(y_true, y_pred)
    return np.sqrt(recall * spec)

# Load and preprocess the data
preprocessed_data = pd.read_csv('./preprocessing_st_data.csv')
eeg_columns = preprocessed_data.drop(['Participant id', 'song_label', 'class','Time'], axis=1).columns

# Group by 'Participant id' and 'song_label'
grouped = preprocessed_data.groupby(['Participant id', 'song_label'])

# Convert each group to a tensor and store them in a list
sequences = [torch.tensor(group[eeg_columns].values, dtype=torch.float32) for _, group in grouped]
padded_sequences = pad_sequence(sequences, batch_first=True)
labels = torch.tensor(preprocessed_data.groupby(['Participant id', 'song_label'])['class'].first().values, dtype=torch.long)

SEED = 42  # Or any other integer
torch.manual_seed(SEED)

# Combine and split datasets
combined_dataset = TensorDataset(padded_sequences, labels)
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])


# Create data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Hyperparameters and model instantiation
input_dim = padded_sequences.shape[2]
hidden_dim = 64
num_layers = 2
output_dim = len(labels.unique())
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
early_stopping = EarlyStopping(patience=5, verbose=True)

num_epochs = 15
for epoch in range(num_epochs):
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

    # Early Stopping
    val_loss = loss.item()  # Assuming you don't have a separate validation set, otherwise compute validation loss
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # Update learning rate
    scheduler.step(loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(y_batch.numpy())

accuracy = 100 * correct / total
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
g_mean = geometric_mean(all_labels, all_preds)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")
print(f"Geometric Mean: {g_mean * 100:.2f}%")
