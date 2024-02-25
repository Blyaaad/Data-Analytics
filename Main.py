import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx

# Read the CSV file into a pandas DataFrame
cyber_sec_atk = pd.read_csv('C:\Analytics\cybersecurity_attacks-cleaned.csv')
#depending on your file location

# Set display options to show all columns and limit rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print(cyber_sec_atk.head(1001))

#Single Variable tables and charts
#Frequncy distribution table
freq_column = input("What column would you want in the table? (\n*Protocol "
                   "\n*Packet Type \n*Traffic Type \n*Malware Indicators \n*Anomaly Scores "
                   "\n*Attack Type \n*Attack Signature \n*Action Taken \n*Severity \n*Device Information "
                   "\n*Network Segment \n*Firewall Logs \n*IDS/IPS Alerts \n*Log Source\n)\n")
frequency_table = cyber_sec_atk[freq_column].value_counts().reset_index()
frequency_table.columns = ['Value', 'Frequency']
# Sort the table based on the values (optional)
frequency_table = frequency_table.sort_values(by='Value')
print(frequency_table)

# Menu for Map Type
while True:
    print("\nChoose a map type:")
    print("1. Bar Chart")
    print("2. Pie Chart")
    print("3. Histogram")
    print("4. Tree Map")
    print("5. Choropleth Map")
    print("0. Exit")

    choice = input("Enter your choice (0-5): ")

    if choice == '0':
        print("Exiting the program.")
        break
    elif choice == '1':
        # Bar Chart
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
    elif choice == '2':
        # Pie Chart
        # Prompt the user to input the column name
        column_name = input("Enter the column name for which you want to create a pie chart: ")

        # Check if the entered column name exists in the DataFrame
        if column_name not in cyber_sec_atk.columns:
            print("Column '{}' not found in the DataFrame.".format(column_name))
        else:
            # Count the frequency of each category in the specified column
            value_counts = cyber_sec_atk[column_name].value_counts()

            # Plotting
            plt.figure(figsize=(8, 8))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Pie Chart of {}'.format(column_name))
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.show()

    elif choice == '3':
        # Histogram
        # Prompt the user to input the column name
        column_name = input("Enter the column name for which you want to create a histogram: ")

        # Check if the entered column name exists in the DataFrame
        if column_name not in cyber_sec_atk.columns:
            print("Column '{}' not found in the DataFrame.".format(column_name))
        else:
            # Plotting
            plt.figure(figsize=(10, 6))
            n, bins, patches = plt.hist(cyber_sec_atk[column_name], bins='auto', color='skyblue',
                                        edgecolor='black')  # 'auto' selects the number of bins automatically
            plt.title('Histogram of {}'.format(column_name))
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.grid(True)

            # Add frequency on top of each bar
            for i, freq in enumerate(n):
                if freq != 0:
                    plt.text(bins[i] + (bins[i + 1] - bins[i]) / 2, freq, str(int(freq)), ha='center', va='bottom')

            plt.show()
    elif choice == '4':
        # Tree Map
        # Prompt the user to input the column name
        column_name = input("Enter the column name for which you want to create a tree map: ")

        # Check if the entered column name exists in the DataFrame
        if column_name not in cyber_sec_atk.columns:
            print("Column '{}' not found in the DataFrame.".format(column_name))
        else:
            # Count the frequency of each category in the specified column
            value_counts = cyber_sec_atk[column_name].value_counts().reset_index()
            value_counts.columns = ['Category', 'Frequency']

            # Plotting Tree Map
            fig = px.treemap(value_counts, path=['Category'], values='Frequency',
                             title='Tree Map of {}'.format(column_name))
            fig.show()
    elif choice == '5':
        # Choropleth Map
        country_column = input("Enter the column name for which you want to create a Choropleth Map: ")
        value_column = input("Enter the column name for numerical values for the Choropleth Map: ")
        if country_column in cyber_sec_atk.columns and value_column in cyber_sec_atk.columns:
            fig = px.choropleth(
                cyber_sec_atk,
                locations=country_column,
                locationmode='country names',
                color=value_column,
                color_continuous_scale='Viridis',
                title=f'Choropleth Map of {value_column} by {country_column}'
            )
            fig.update_layout(geo=dict(showcoastlines=True, showland=True), coloraxis_colorbar=dict(title=value_column))
            fig.show()
        else:
            print("Columns not found in the DataFrame.")
    else:
        print("Invalid choice. Please enter a number between 0 and 5.")
