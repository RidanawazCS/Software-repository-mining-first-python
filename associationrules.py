import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Read the CSV file into a DataFrame
csv_file_path = 'output_selected_columns.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file_path)

# Check if 'Processed_description' column exists in the dataset
if 'Processed_Description' not in df.columns:
    print("Error: 'Processed_Description' column not found in the dataset.")
else:
    # Select the first 100 rows and 'Processed_description' column
    selected_data = df.loc[:99, 'Processed_Description'].astype(str)

    # Split the processed data into lists of transactions
    transactions = [item.split() for item in selected_data]

    # Use TransactionEncoder to convert the transaction data into a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent itemsets using Apriori algorithm
    frequent_itemsets = apriori(one_hot_df, min_support=0.1, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Display the association rules
    print("Association Rules:")
    print(rules)

    # Save association rules to a text file
    output_file_path = 'association_rules.txt'  # Replace with your desired output file path

    with open(output_file_path, 'w') as output_file:
        output_file.write("Association Rules:\n")
        output_file.write(rules.to_string(index=False))

    print(f"Association rules saved to {output_file_path}")