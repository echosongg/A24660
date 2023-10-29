import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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


# Given the ground truth labels and predictions, this function calculates and prints the specified metrics
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    g_mean = geometric_mean(y_true, y_pred)

    print(f"Evaluation metrics for {model_name}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Geometric Mean: {g_mean * 100:.2f}%")
    print("-----------------------------")
# 1. 加载数据
music_data = pd.read_csv('./preprocessing_st_data.csv')

# 2. 使用特征选择的掩码
features = music_data.drop(["Participant id", "class","song_label","Time"], axis=1).columns


# 3. 根据选择的特征更新数据
X = music_data[features]
y = music_data["class"]

# Splitting the data, training the models, and making predictions as provided in the original code

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

# Train Logistic Regression
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)

# Train SVM
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

# Now, evaluate the models
evaluate_model(y_test, y_pred_rf, "RandomForest Classifier")
evaluate_model(y_test, y_pred_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_svm, "SVM")




