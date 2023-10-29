# A24660
## code run order
1. concat_dataset.py: combine all raw data.
2. data_preprocessing.py: deal with NaN, 0 value, and min-max normalize, and median filter
3. statistic_feature_calculation.py: calculate the statistic feature from raw data, such as min_EEG, max_EEG
4. LSTM.py: LSTM model
5. CNN.py: CNN model
6. LSTM_best_para_combo.py: Find the best hyperparameter combination for LSTM model
7. machine_learning_classification.py: Use random forest, SVM, logistic regression model classify preprocessed EEG data.
8. visualize.py: visualize the result, used in paper.

## data
Raw data from "Use music-affect_v2-eeg-timeseries"
combined.csv from concat_dataset.py.
preprocessing_data.csv from data_preprocessing.py.
preprocessing_st_data.csv from statistic_feature_calculation.py.

