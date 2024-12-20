import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string

# Download NLTK stopwords data (run once)
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords_and_save(csv_file):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Check if 'Description' column exists in the dataset
        if 'Description' not in df.columns:
            print("Error: 'Description' column not found in the dataset.")
            return

        # Remove stopwords from the 'Description' column
        stop_words = set(stopwords.words('english'))
        df['Processed_Description'] = df['Description'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(str(x)) if word.lower() not in stop_words and word.lower() not in string.punctuation]))

        # Save the DataFrame with processed data to the same CSV file
        df.to_csv(csv_file, index=False)

        print(f"Stopwords removed and processed data saved to {csv_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
csv_file_path = 'output_selected_columns.csv'  # Replace with the path to your input CSV file
remove_stopwords_and_save(csv_file_path)