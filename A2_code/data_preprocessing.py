import pandas as pd
data = pd.read_csv('./combined.csv')
# Apply min-max normalization for each participant's EEG data
eeg_columns = ['EEG01', 'EEG02', 'EEG03', 'EEG04', 'EEG05', 'EEG06', 'EEG07', 'EEG08']

# Apply linear interpolation to handle NaN values and 0 values
def interpolate_values(group):
    """Apply linear interpolation for each EEG column in the group to handle NaNs and 0 values."""
    # Handle NaN values
    group[eeg_columns] = group[eeg_columns].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    # Handle 0 values
    group[eeg_columns] = group[eeg_columns].replace(0, pd.NA).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return group

def normalize_eeg(group):
    """Apply min-max normalization for each EEG column in the group."""
    for column in eeg_columns:
        group[column] = (group[column] - group[column].min()) / (group[column].max() - group[column].min())
    return group
# Apply median smoothing filter for each participant's normalized EEG data
def median_filter(group):
    """Apply median smoothing filter for each EEG column in the group."""
    for column in eeg_columns:
        group[column] = group[column].rolling(window=3, center=True).median().fillna(method='bfill').fillna(method='ffill')
    return group
processed_data = data.groupby('Participant id').apply(interpolate_values)

normalized_data = processed_data.groupby(['Participant id', 'song_label']).apply(normalize_eeg)

smoothed_data = normalized_data.groupby(['Participant id', 'song_label']).apply(median_filter)

# Saving the combined reshaped data to CSV
smoothed_data_file_path = "./preprocessing_data.csv"
smoothed_data.to_csv(smoothed_data_file_path, index=False)

print("save over")