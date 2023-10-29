import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

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

# 定义一个函数来计算几何均值
def geometric_mean(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='macro')
    spec = specificity(y_true, y_pred)
    return np.sqrt(recall * spec)
preprocessed_data = pd.read_csv('./preprocessing_st_data.csv')
eeg_columns = preprocessed_data.drop(['Participant id', 'song_label', 'class', 'Time'], axis=1).columns
X = preprocessed_data[eeg_columns].values
y = preprocessed_data['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)


class EEGNet(nn.Module):
    def __init__(self, num_classes):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Early Stopping Class [Same as before]
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

print('start training')
model = EEGNet(len(set(y_train)))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=5, verbose=True)

for epoch in range(15):
    model.train()  # Set model to training mode
    running_loss = 0.0
    for batch_X, batch_y in DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(X_train_tensor)
    print(f"Epoch [{epoch + 1}/15] Loss: {avg_train_loss:.4f}")

    early_stopping(avg_train_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print('start testing')
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
all_preds = []
all_labels = []

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

with torch.no_grad():
    for batch_X, batch_y in DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False):
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

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