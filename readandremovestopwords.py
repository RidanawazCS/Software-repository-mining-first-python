import pandas as pd
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK tokenization data (run once)
nltk.download('punkt')

def read_and_tokenize_description(csv_file):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if 'Description' column exists in the dataset
        if 'Description' not in df.columns:
            print("Error: 'Description' column not found in the dataset.")
            return

        # Fill NaN values in the 'Description' column with an empty string
        df['Description'].fillna('', inplace=True)

        # Get the 'Description' column as a list
        description_list = df['Description'].tolist()

        # Tokenize the 'Description' content
        tokenized_description = [word_tokenize(str(description)) for description in description_list]

        # Print the tokenized content
        print("Tokenized Description:")
        for tokens in tokenized_description:
            print(tokens)

        return tokenized_description

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage
csv_file_path = 'output_selected_columns.csv'  # Replace with the path to your CSV file
tokenized_description_list = read_and_tokenize_description(csv_file_path)