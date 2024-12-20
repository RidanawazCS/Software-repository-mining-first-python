import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Read the dataset from the CSV file
file_path = 'mozilla_firefox.csv'
df = pd.read_csv(file_path)

# Assuming 'description' is the column containing textual data
text_data = df['Description'].astype(str)

# Step 2: Apply TfidfVectorizer to convert text data to feature vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Adjust parameters as needed
feature_matrix = vectorizer.fit_transform(text_data)

# Step 3: Convert the sparse matrix to a dense array if needed
dense_feature_matrix = feature_matrix.toarray()

# Step 4: Concatenate the dense feature matrix with the original DataFrame
df = pd.concat([df, pd.DataFrame(dense_feature_matrix, columns=vectorizer.get_feature_names_out())], axis=1)

# Step 5: Print or save the updated DataFrame
print(df.head())

# Optionally, you can save the updated DataFrame to a new CSV file
df.to_csv('updated_dataset.csv', index=False)