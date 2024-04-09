from pyspark.sql.functions import col
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
    plot_df = null_counts_df.filter(F.col('%null') > 0).toPandas()
    
    return plot_df

def unique_check(df, column,spark):
    
    if df.select(col(column)).distinct().count()== df.count():
        return(f"Column {column} contains unique values")
    else:
        return(f"Column {column} does not contain unique values")


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



    