from pyspark.sql.functions import *
from pyspark.sql import functions as F
import matplotlib.pyplot as plt

def nulls_buster(df, spark):
    
    # Create a list of tuples containing column name and percentage of null or empty values
    null_counts = [(col_name,
                    df.filter((F.col(col_name).isNull()) | (F.col(col_name) == '')).count() / df.count() * 100)
                   for col_name in df.columns]
    
    # Create a DataFrame from the list of tuples
    null_counts_df = spark.createDataFrame(null_counts, ['column_name', '%null'])
    
    # Filter the DataFrame to include only rows where percentage of null values is greater than 0
    plot_df = null_counts_df.filter(F.col('%null') > 0).orderBy('%null', ascending=False).toPandas()

    
    return plot_df
#########################################################
#########################################################

def unique_check(df, column,spark):
    
    if df.select(col(column)).distinct().count()== df.count():
        return(f"Column {column} contains unique values")
    else:
        return(f"Column {column} does not contain unique values")

#########################################################
#########################################################

def nulls_buster_visual(df,spark):
    from pyspark.sql import functions as F
    import matplotlib.pyplot as plt
    
    # Create a list of tuples containing column name and count of null or empty values
    null_counts = [(col_name,
                    df.filter((F.col(col_name).isNull()) | (F.col(col_name) == '')).count()/df.count()*100)
                   for col_name in df.columns]
    
    
    null_counts_df = spark.createDataFrame(null_counts, ['column_name', '%null'])
    
    # Filter the DataFrame to include only rows where count is greater than 0
    plot_df = null_counts_df.filter(F.col('%null') > 0).toPandas()
    
    # Plot the bar chart
    plt.figure(figsize=(12, 6))  # Adjust width as needed, height is optional
    plt.bar(plot_df['column_name'], plot_df['%null'])
    plt.xlabel('Column Name')
    plt.ylabel('Null%')
    plt.title('Counts of Null and Whitespace Values')
    plt.xticks(rotation=45)
    return plt.show()

#######################################################
#######################################################
import os

def split_csv(input_file, max_chunk_size_mb=25):
    """Splits a CSV file into multiple smaller chunks, preserving row integrity.

    Args:
        input_file (str): Path to the input CSV file.
        max_chunk_size_mb (int, optional): Maximum chunk size in megabytes. Defaults to 25.
    """

    max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
    current_chunk = []
    current_chunk_size = 0
    chunk_number = 1

    with open(input_file, 'r') as input_file:  # Open in text mode for CSV
        for line in input_file:
            line_length = len(line.encode())  # Calculate byte length accurately
            if current_chunk_size + line_length > max_chunk_size_bytes:
                output_file = f'{os.path.splitext(input_file.name)[0]}_part{chunk_number}.csv'
                with open(output_file, 'w') as output_file:  # Open in text mode for CSV
                    output_file.writelines(current_chunk)
                print(f'Saved chunk {chunk_number} to {output_file}')
                chunk_number += 1
                current_chunk = []
                current_chunk_size = 0
            current_chunk.append(line)
            current_chunk_size += line_length

        # Write the last chunk if any
        if current_chunk:
            output_file = f'{os.path.splitext(input_file.name)[0]}_part{chunk_number}.csv'
            with open(output_file, 'w') as output_file:  # Open in text mode for CSV
                output_file.writelines(current_chunk)
            print(f'Saved chunk {chunk_number} to {output_file}')

#######################################################
#######################################################
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def plot_top_n_bycolumn(df, column,spark, n=10):
    """
    Plot the top N crime code descriptions by count.

    Args:
        df (DataFrame): PySpark DataFrame containing crime data.
        column (str): Name of the column containing crime code descriptions.
        n (int): Number of top crime code descriptions to plot. Defaults to 10.

    Returns:
        None
    """

    # Perform aggregation with correct column usage
    agg_df = df.groupBy(column).agg(count('*').alias('Count'))

    # Sort and select top N using F.col for column reference
    top_n_df = agg_df.orderBy(F.col('Count').desc()).limit(n)

    # Convert to Pandas DataFrame and plot with correct column references
    top_n_pandas_df = top_n_df.toPandas()
    plt.figure(figsize=(12, 5))
    top_n_pandas_df.plot(kind='bar', x=column, y='Count', legend=None)
    plt.xlabel(column)  # Use dynamic column name
    plt.ylabel('Count')
    plt.title(f'Top {n} {column} by Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()  # Adjust layout for better readability
    plt.show();  # Display the plot

    