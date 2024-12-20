import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Read the updated dataset
file_path_updated = 'updated_dataset.csv'
df_updated = pd.read_csv(file_path_updated)

# Step 2: Handle missing or NaN values
# For numeric columns, replace missing values with the mean of the column
numeric_columns = df_updated.select_dtypes(include=['float64', 'int64']).columns
df_updated[numeric_columns] = df_updated[numeric_columns].fillna(df_updated[numeric_columns].mean())

# For categorical columns, replace missing values with a placeholder (e.g., 'missing')
categorical_columns = df_updated.select_dtypes(include=['object']).columns
df_updated[categorical_columns] = df_updated[categorical_columns].fillna('missing')

# Step 3: Apply label encoding for categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    df_updated[col] = label_encoder.fit_transform(df_updated[col])

# Step 4: Split data into features (X) and target variable (y)
X = df_updated.drop('Resolution', axis=1)  # Replace 'target_column_name' with your target column
y = df_updated['Resolution']  # Replace 'target_column_name' with your target column

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Initialize and fit the HistGradientBoostingClassifier
hgbc_classifier = HistGradientBoostingClassifier()
hgbc_classifier.fit(X_train, y_train)

# Step 7: Make predictions on the test set
predictions = hgbc_classifier.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of HistGradientBoostingClassifier: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

# Save results to CSV file
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': predictions
})
results_df.to_csv('hist_gradient_boosting_results.csv', index=False)