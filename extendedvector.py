from sklearn.feature_extraction.text import CountVectorizer
import string

def calculate_and_store_bow_vector(input_txt, output_txt):
    try:
        # Read the content from the input text file
        with open(input_txt, 'r', encoding='utf-8') as file:
            content = file.read()

        # Tokenize the content
        tokens = content.split()  # Splitting by space for simplicity; you can use a more sophisticated tokenizer

        # Remove punctuation and convert to lowercase
        tokens = [word.lower().strip(string.punctuation) for word in tokens]

        # Create a CountVectorizer
        vectorizer = CountVectorizer()

        # Fit and transform the content using the Bag of Words model
        bow_matrix = vectorizer.fit_transform(tokens)

        # Convert the BoW matrix to a dense array
        bow_array = bow_matrix.toarray()

        # Convert the array to a string and store it in another text file
        bow_vector_str = '\n'.join([' '.join(map(str, row)) for row in bow_array])

        with open(output_txt, 'w', encoding='utf-8') as output_file:
            output_file.write(bow_vector_str)

        print(f"Extended vector using Bag of Words saved to {output_txt}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
input_txt_file = 'stopwordsremoved.txt'  # Replace with the path to your input text file
output_txt_file = 'output_bow_vector.txt'  # Replace with the path for the output text file

calculate_and_store_bow_vector(input_txt_file, output_txt_file)