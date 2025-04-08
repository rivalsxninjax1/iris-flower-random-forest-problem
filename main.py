import numpy as np 
import pandas as pd  # For data handling
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier
from sklearn.metrics import accuracy_score  # To calculate model accuracy
import matplotlib.pyplot as plt  # For plotting the information

# Loading the dataset from the CSV file into a Pandas DataFrame
df = pd.read_csv('data/iris.csv')

# Displaying the first few rows to verify if the program is working correctly
print('Dataset Preview:')
print(df.head())

# Separating the features and the target variable
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']  # Target variable: species

# Splitting the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating the Random Forest model with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model using the training data
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the random forest model: {accuracy * 100:.2f}%")

# Plotting the feature importances
importances = model.feature_importances_  # Importance of each feature from the model
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Creating a bar plot for visualization of feature importances
plt.figure(figsize=(8, 5))
plt.bar(feature_names, importances, color='skyblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest Model")
plt.show()

