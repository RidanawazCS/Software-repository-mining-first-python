from collections import Counter
import string

def calculate_and_store_support(input_txt, output_txt):
    try:
        # Read the content from the input text file
        with open(input_txt, 'r', encoding='utf-8') as file:
            content = file.read()

        # Tokenize the content and remove punctuation
        tokens = [word.lower().strip(string.punctuation) for word in content.split()]

        # Calculate support using Counter
        token_counts = Counter(tokens)
        total_tokens = len(tokens)

        # Calculate support values
        support_values = {token: count / total_tokens for token, count in token_counts.items()}

        # Sort tokens by support values in descending order
        sorted_tokens = sorted(support_values.items(), key=lambda x: x[1], reverse=True)

        # Create a string with tokens and corresponding support values
        result_str = '\n'.join([f'{token}: {support:.4f}' for token, support in sorted_tokens])

        # Store the result in another text file
        with open(output_txt, 'w', encoding='utf-8') as output_file:
            output_file.write(result_str)

        print(f"Support values calculated and saved to {output_txt}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
input_txt_file = 'stopwordsremoved.txt'  # Replace with the path to your input text file
output_txt_file = 'output_support_values.txt'  # Replace with the path for the output text file

calculate_and_store_support(input_txt_file, output_txt_file)