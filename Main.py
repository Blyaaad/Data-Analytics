import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
cyber_sec_atk = pd.read_csv('data/cybersecurity_attacks.csv') #depending on your file location

# Set display options to show all columns and limit rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Display the first 1000 rows of the DataFrame
print(cyber_sec_atk.head(1001))
