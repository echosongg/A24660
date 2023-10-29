import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
from A2_code.LSTM import LSTMModel  # Assuming you have an LSTM model defined in a separate module

# Load and preprocess the data

# Load the preprocessed dataset
preprocessed_data = pd.read_csv('./preprocessing_st_data.csv')

# Extract EEG columns (excluding non-EEG columns)
eeg_columns = preprocessed_data.drop(['Participant id', 'song_label', 'class', 'Time'], axis=1).columns

# Group the data by 'Participant id' and 'song_label'
grouped = preprocessed_data.groupby(['Participant id', 'song_label'])

# Convert each group to a tensor and store them in a list
sequences = [torch.tensor(group[eeg_columns].values, dtype=torch.float32) for _, group in grouped]

# Pad the sequences so they have the same length
padded_sequences = pad_sequence(sequences, batch_first=True)

# Convert song labels to tensor
labels = torch.tensor(preprocessed_data.groupby(['Participant id', 'song_label'])['class'].first().values,
                      dtype=torch.long)

# Create a dataset combining the padded sequences and labels
combined_dataset = TensorDataset(padded_sequences, labels)

# Split the dataset into 80% training and 20% testing sets
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# Model definition

# Hyperparameters for grid search
input_dim = padded_sequences.shape[2]
output_dim = len(labels.unique())
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [4, 8, 16]
hidden_dims = [64, 128]
num_layers_list = [1, 2, 3]
weight_decays = [1e-5, 1e-4]
num_epochs = 10

# Placeholders for the best model parameters and its accuracy
best_params = {}
best_accuracy = 0

# Grid search over hyperparameters
for lr in learning_rates:
    for batch_size in batch_sizes:
        for hidden_dim in hidden_dims:
            for num_layers in num_layers_list:
                for weight_decay in weight_decays:

                    # Create DataLoader with the current batch size
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                    # Initialize the LSTM model with the current hyperparameters
                    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

                    # Train the model
                    for epoch in range(num_epochs):
                        for X_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            loss.backward()
                            optimizer.step()
                        scheduler.step(loss)

                    # Evaluate the model
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for X_batch, y_batch in test_loader:
                            outputs = model(X_batch)
                            _, predicted = torch.max(outputs.data, 1)
                            total += y_batch.size(0)
                            correct += (predicted == y_batch).sum().item()
                    accuracy = 100 * correct / total

                    # Update the best accuracy and hyperparameters if the current model is better
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            "learning_rate": lr,
                            "batch_size": batch_size,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "weight_decay": weight_decay
                        }

# Print the best hyperparameters and their accuracy
print("Best Parameters:", best_params)
print("Best Accuracy:", best_accuracy)