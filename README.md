# The PySpark Chronicles: Adventures in Data Processing

![Data Exploration](castle.jpeg)
<br>
*Picture created with Hugging Face Stable Diffusion*
[Huggingface](https://huggingface.co/spaces/stabilityai/stable-diffusion)
<br>
# Exploring the Enchanted World of PySpark

Welcome to the "The PySpark Chronicles: Exploring Data Processing Adventures" project! Here, we embark on an exciting journey into the dynamic realm of PySpark, an intuitive Python API designed for Apache Spark. We try uncovering a multitude of data exploration, transformation endeavors and even take a sneak peak into ML libraries. 
## Overview

Apache Spark is a fast and general-purpose cluster computing system that provides high-level APIs in Java, Scala, and Python, and an optimized engine that supports general execution graphs. PySpark, the Python API for Spark, allows us to interact with Spark using Python.

This project delves into the analysis of a crime dataset available at [Crime Data](https://catalog.data.gov/dataset/crime-data-from-2020-to-present/resource/5eb6507e-fa82-4595-a604-023f8a326099), which consists of 925,720 records and 28 columns. The aim is to provide a beginner-friendly guide to using PySpark, concentrating on elementary commands for data exploration and manipulation. Once the data preprocessing phase is complete, the refined data is utilized to train a logistic regression model. This model attempt to classify crime types as robbery or non-robbery, based on various attributes. The project concludes by evaluating the model's effectiveness and displaying the accuracy of its predictions, offering a glimpse into machine learning to underscore the significance of the preparatory data handling stages.




## Contents

- **PySparkChronicles_Chapter1_DataExploration.ipynb**: Jupyter Notebook containing basic commands for data exploration using PySpark.
- **PySparkChronicles_Chapter2_DataCuration.ipynb**: Jupyter Notebook demonstrating various data transformation operations in PySpark.
- **PySparkChronicles_Chapter3_DataPreprocessingAndML.ipynb**: Jupyter Notebook for processing dataset and encoding your target variable
- **common_libraries.py**: File listing the Python dependencies required to run the project.
- **project_function.py**: File listing all functions required to run the project.
- **crime_data.csv**: File which reflects incidents of crime in the City of Los Angeles dating back to 2020 (data.gov).
- **SparkOverview.md**: Quick notes about Spark Partitions and Infrastructure generated with Google Gemini.

## Getting Started

To run the code in this project, follow these steps:

1. Ensure you have Python and Jupyter Notebook installed on your system.
2. Clone this repository to your local machine.
3. Open the Jupyter Notebooks `PySparkChronicles_Chapter1_DataExploration.ipynb`,`PySparkChronicles_Chapter2_DataCuration.ipynb`, and `PySparkChronicles_Chapter3_DataProcessingAndML.ipynb`,`project_functions.py`,`common_libraries.py` to explore and run the code.

## Python Packages

The project relies on the following Python packages:
```
# Spark Data Processing
from pyspark.sql import SparkSession, DataFrame

# System Interactions
import os
import logging

# Data Manipulation (Spark SQL)
from pyspark.sql import functions as F
from pyspark.sql.functions import *

# Data Manipulation (pyspark.sql.types)
from pyspark.sql.types import *

# Visualization Libraries (Optional)
# Uncomment these lines if you plan to use these libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Spark Machine Learning
from pyspark.ml.feature import StringIndexer, OneHotEncoder
# Comment out the line below if you're not using scikit-learn
# from sklearn.preprocessing import OneHotEncoder

# Spark Machine Learning Classification Algorithms
from pyspark.ml.classification import LinearSVC, LogisticRegression

# Spark Machine Learning Feature Engineering
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# scikit-learn Integration (Optional)
# Uncomment these lines if you plan to use scikit-learn functionalities
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score

# Data Imbalance Handling (Optional)
# Uncomment this line if you plan to use SMOTE
from imblearn.over_sampling import SMOTE

# Spark Machine Learning Feature Scaling (Optional)
#Uncomment this line if you plan to use StandardScaler
from pyspark.ml.feature import StandardScaler

# Spark Machine Learning Evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics```
```
## Acknowledgment
I would like to acknowledge Stackoverflow, ChatGPT, Google Bard as an instrumental aid in the development of this project.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
