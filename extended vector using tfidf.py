from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Read the contents from the input text file
input_txt_path = 'stopwordsremoved.txt'  # Replace with the path to your input text file
with open(input_txt_path, 'r', encoding='utf-8') as file:
    contents = file.readlines()

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the contents to obtain the TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(contents)

# Convert the TF-IDF matrix to a dense array
dense_matrix = tfidf_matrix.toarray()

# Save the extended vector to a new text file
output_txt_path = 'output_extendedtfidf_vector.txt'  # Replace with your desired output file path
np.savetxt(output_txt_path, dense_matrix, delimiter='\t')

print(f"Extended vector saved to {output_txt_path}")