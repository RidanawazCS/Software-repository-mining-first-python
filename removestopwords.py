from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

# Download NLTK tokenization and stopwords data (run once)
nltk.download('punkt')
nltk.download('stopwords')

def process_and_save_tokens(input_txt, output_txt):
    try:
        # Read the content from the input text file
        with open(input_txt, 'r', encoding='utf-8') as file:
            content = file.read()

        # Tokenize the content
        tokens = word_tokenize(content)

        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.lower() not in string.punctuation]

        # Join the processed tokens into a string
        processed_text = ' '.join(tokens)

        # Save the processed tokens to the output text file
        with open(output_txt, 'w', encoding='utf-8') as file:
            file.write(processed_text)

        print(f"Processed tokens saved to {output_txt}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
input_txt_file = 'tokens.txt'  # Replace with the path to your input text file
output_txt_file = 'stopwordsremoved.txt'  # Replace with the path for the output text file

process_and_save_tokens(input_txt_file, output_txt_file)