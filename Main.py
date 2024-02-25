import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
cyber_sec_atk = pd.read_csv('original.csv') #depending on your file location

# Set display options to show all columns and limit rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(cyber_sec_atk.head(1001))

#Single Variable tables and charts
#Frequncy distribution table
freq_column = input("What column would you want in the table? (Protocol, "
                   "Packet Type, Traffic Type, Malware Indicators, Anomaly Scores, "
                   "Attack Type, Attack Signature, Action Taken, Severity, Device Information, "
                   "Network Segment, Firewall Logs, IDS/IPS Alerts, Log Source)")
frequency_table = cyber_sec_atk[freq_column].value_counts().reset_index()
frequency_table.columns = ['Value', 'Frequency']
# Sort the table based on the values (optional)
frequency_table = frequency_table.sort_values(by='Value')
print(frequency_table)

#Bar chart
#Change the 'x_column' and 'y_column' to the columns that we want to plot
# Set display options to show all columns and limit rows
# Prompt the user to input the column name
column_name = input("Enter the column name for which you want to create a frequency distribution: ")

# Group by the x column and count the occurrences
grouped = cyber_sec_atk.groupby(column_name).size()
# Plotting
# Change the labels to the appropriate terms
ax = grouped.plot(kind='bar')
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.title('Bar Chart of Frequency Grouped by ' + column_name)

# Add labels on top of each bar
for i, freq in enumerate(grouped):
    ax.text(i, freq + 0.05, str(freq), ha='center', va='bottom')

plt.show()



