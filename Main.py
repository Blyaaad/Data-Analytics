import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
cyber_sec_atk = pd.read_csv('original.csv') #depending on your file location

# Set display options to show all columns and limit rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(cyber_sec_atk.head(1001))

column_name = input("What column would you want in the table? (Protocol, "
                   "Packet Type, Traffic Type, Malware Indicators, Anomaly Scores, "
                   "Attack Type, Attack Signature, Action Taken, Severity, Device Information, "
                   "Network Segment, Firewall Logs, IDS/IPS Alerts, Log Source)")

#Single Variable tables and charts
#Frequncy distribution table
frequency_table = cyber_sec_atk[column_name].value_counts().reset_index()
frequency_table.columns = ['Value', 'Frequency']

# Sort the table based on the values (optional)
frequency_table = frequency_table.sort_values(by='Value')

print(frequency_table)


'''
#Bar chart
#Change the 'x_column' and 'y_column' to the columns that we want to plot
x_column = 'x_column'
y_column = 'y_column'

# Group by the x column and sum the y column
grouped = cyber_sec_atk.groupby(x_column)[y_column].sum()

# Plotting
# Change the labels to the appropriate terms
grouped.plot(kind='bar')
plt.xlabel(x_column)
plt.ylabel('Sum of ' + y_column)
plt.title('Bar Chart of ' + y_column + ' Grouped by ' + x_column)
plt.show()
'''

