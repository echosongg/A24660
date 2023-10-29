import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv('./preprocessing_st_data.csv')

# Extract features and target variable
X = data.drop(columns=['Time', 'Participant id', 'class','song_label'])
y = data['class']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)

# Train the model
mlp.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(accuracy,report)
