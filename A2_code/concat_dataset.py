import pandas as pd
import numpy as np
# File paths and corresponding labels
file_paths = [f"./music-affect_v2/music-affect_v2-eeg-timeseries/S{i}.xlsx" for i in range(1, 13)]
labels = [0] * 4 + [1] * 4 + [2] * 4
song_labels = np.arange(1,13)


def process_and_reshape_file_corrected(file_path, song_class, song_label):
    # Load the data
    df = pd.read_excel(file_path)

    # Reshape the data: every 8 rows should become 8 columns
    reshaped_data = []

    for col in df.columns[1:]:  # Start from 1 to skip the "Time" column
        for i in range(0, len(df), 8):
            subset = df.iloc[i:i + 8]
            eeg_values = subset[col].values
            time_values = subset["Time"].values[0]
            reshaped_data.append([time_values, col] + list(eeg_values))

    # Convert to DataFrame
    reshaped_df = pd.DataFrame(reshaped_data,
                               columns=["Time", "Participant id"] + [f"EEG{i + 1:02}" for i in range(8)])
    reshaped_df['class'] = song_class
    # Add song_label column
    reshaped_df["song_label"] = song_label

    return reshaped_df

# Loading all reshaped datasets and adding labels
reshaped_datasets = []
for file_path, label, song_label in zip(file_paths, labels, song_labels):
    reshaped_data = process_and_reshape_file_corrected(file_path, label, song_label)
    reshaped_datasets.append(reshaped_data)

# Concatenating all reshaped datasets into one
combined_reshaped_data = pd.concat(reshaped_datasets, axis=0, ignore_index=True)

# Saving the combined reshaped data to CSV
combined_reshaped_data_file_path = "./combined.csv"
combined_reshaped_data.to_csv(combined_reshaped_data_file_path, index=False)

print("save over")