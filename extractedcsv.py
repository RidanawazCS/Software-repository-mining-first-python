import pandas as pd
def extract_and_create_csv(input_csv, output_csv):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_csv)

        # Check if 'Title' and 'Description' columns exist in the dataset
        if 'Title' not in df.columns or 'Description' not in df.columns:
            print("Error: 'Title' or 'Description' columns not found in the dataset.")
            return

        # Select 'Title' and 'Description' columns
        selected_columns = df[['Title', 'Description']]

        # Save the selected columns to a new CSV file
        selected_columns.to_csv(output_csv, index=False)

        print(f"Selected columns saved to {output_csv}")

    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
input_csv_file = 'mozilla_firefox.csv'  # Replace with the path to your input CSV file
output_csv_file = 'output_selected_columns.csv'  # Replace with the path for the output CSV file

extract_and_create_csv(input_csv_file, output_csv_file)