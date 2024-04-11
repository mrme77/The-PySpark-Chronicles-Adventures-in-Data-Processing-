from pyspark.sql.functions import *
from pyspark.sql import functions as F
import matplotlib.pyplot as plt
import os

def nulls_buster(df, spark):
    """
    Calculate the percentage of null or empty values for each column in a DataFrame.

    Args:
        df (DataFrame): PySpark DataFrame to analyze.
        spark (SparkSession): Active SparkSession.

    Returns:
        DataFrame: A Pandas DataFrame containing columns and their corresponding percentages of null values.
    """
    null_counts = [(col_name,
                    df.filter((F.col(col_name).isNull()) | (F.col(col_name) == '')).count() / df.count() * 100)
                   for col_name in df.columns]
    null_counts_df = spark.createDataFrame(null_counts, ['column_name', '%null'])
    plot_df = null_counts_df.filter(F.col('%null') > 0).orderBy('%null', ascending=False).toPandas()
    return plot_df

def unique_check(df, column):
    """
    Check if a column in a DataFrame contains unique values.

    Args:
        df (DataFrame): PySpark DataFrame to check.
        column (str): Name of the column to check for uniqueness.

    Returns:
        str: A message indicating whether the column contains unique values or not.
    """
    if df.select(col(column)).distinct().count() == df.count():
        return f"Column {column} contains unique values"
    else:
        return f"Column {column} does not contain unique values"

def nulls_buster_visual(df, spark):
    """
    Visualize the percentage of null or empty values for each column in a DataFrame.

    Args:
        df (DataFrame): PySpark DataFrame to analyze.
        spark (SparkSession): Active SparkSession.

    Returns:
        None
    """
    null_counts = [(col_name,
                    df.filter((F.col(col_name).isNull()) | (F.col(col_name) == '')).count()/df.count()*100)
                   for col_name in df.columns]
    null_counts_df = spark.createDataFrame(null_counts, ['column_name', '%null'])
    plot_df = null_counts_df.filter(F.col('%null') > 0).toPandas()
    plt.figure(figsize=(12, 6))
    plt.bar(plot_df['column_name'], plot_df['%null'])
    plt.xlabel('Column Name')
    plt.ylabel('Null%')
    plt.title('Counts of Null and Whitespace Values')
    plt.xticks(rotation=45)
    return plt.show()

def split_csv(input_file, max_chunk_size_mb=25):
    """
    Split a CSV file into multiple smaller chunks while preserving row integrity.

    Args:
        input_file (str): Path to the input CSV file.
        max_chunk_size_mb (int, optional): Maximum chunk size in megabytes. Defaults to 25.

    Returns:
        None
    """
    max_chunk_size_bytes = max_chunk_size_mb * 1024 * 1024
    current_chunk = []
    current_chunk_size = 0
    chunk_number = 1

    with open(input_file, 'r') as input_file:
        for line in input_file:
            line_length = len(line.encode())
            if current_chunk_size + line_length > max_chunk_size_bytes:
                output_file = f'{os.path.splitext(input_file.name)[0]}_part{chunk_number}.csv'
                with open(output_file, 'w') as output_file:
                    output_file.writelines(current_chunk)
                print(f'Saved chunk {chunk_number} to {output_file}')
                chunk_number += 1
                current_chunk = []
                current_chunk_size = 0
            current_chunk.append(line)
            current_chunk_size += line_length

        if current_chunk:
            output_file = f'{os.path.splitext(input_file.name)[0]}_part{chunk_number}.csv'
            with open(output_file, 'w') as output_file:
                output_file.writelines(current_chunk)
            print(f'Saved chunk {chunk_number} to {output_file}')

def plot_top_n_bycolumn(df, column, n=10):
    """
    Plot the top N column series descriptions by count.

    Args:
        df (DataFrame): PySpark DataFrame containing crime data.
        column (str): Name of the column containing crime code descriptions.
        n (int): Number of top crime code descriptions to plot. Defaults to 10.

    Returns:
        None
    """
    agg_df = df.groupBy(column).agg(count('*').alias('Count'))
    top_n_df = agg_df.orderBy(F.col('Count').desc()).limit(n)
    top_n_pandas_df = top_n_df.toPandas()
    plt.figure(figsize=(12, 5))
    top_n_pandas_df.plot(kind='bar', x=column, y='Count', legend=None)
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Top {n} {column} by Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def impute_missing_values(data, columns):
    """
    Impute missing values for categorical values in the specified columns with the most frequent value in each column.

    Args:
        data (DataFrame): PySpark DataFrame to impute missing values.
        columns (list): List of column names to process.

    Returns:
        DataFrame: PySpark DataFrame with missing values imputed.
    """
    most_frequent_values = {}
    for col_name in columns:
        most_frequent_values[col_name] = data.filter(col(col_name).isNotNull()) \
                                              .groupBy(col_name).count() \
                                              .orderBy('count', ascending=False) \
                                              .first()[col_name]

    for col_name in columns:
        data = data.withColumn(col_name, when((col(col_name).isNull()) | (col(col_name) == ""), 
                                              most_frequent_values[col_name]).otherwise(col(col_name)))
    
    return data
