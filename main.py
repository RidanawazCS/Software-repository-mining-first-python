import pandas as pd
def read_file(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage
file_path = 'mozilla_firefox.csv'

# Read the CSV file
dt_frame = read_file(file_path)

# Check if the reading was successful
if dt_frame is not None:
    # Display the first few rows of the DataFrame
    print("Data from CSV file:")
    print(dt_frame.head())
else:
    print("Failed to read the CSV file.")