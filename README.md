# The PySpark Chronicles: Adventures in Data Processing

![Data Exploration](castle.jpeg)
<br>
*Picture created with Hugging Face Stable Diffusion*
[Huggingface](https://huggingface.co/spaces/stabilityai/stable-diffusion)
<br>
# Exploring the Enchanted World of PySpark

Welcome to the "The PySpark Chronicles: Exploring Data Processing Adventures" project! Here, we embark on an exciting journey into the dynamic realm of PySpark, an intuitive Python API designed for Apache Spark. We try uncovering a multitude of data exploration, transformation endeavors and even take a sneak peak into ML classification task.

## Overview

Apache Spark is a fast and general-purpose cluster computing system that provides high-level APIs in Java, Scala, and Python, and an optimized engine that supports general execution graphs. PySpark, the Python API for Spark, allows us to interact with Spark using Python.

This project aims to provide a beginner-friendly introduction to PySpark, focusing on basic commands for data exploration and transformations.

## Contents

- **data_exploration.ipynb**: Jupyter Notebook containing basic commands for data exploration using PySpark.
- **data_transformations.ipynb**: Jupyter Notebook demonstrating various data transformation operations in PySpark.
- **data_preprocessing.ipynb**: Jupyter Notebook for processing dataset and encoding your target variable
- **common_libraries.py**: File listing the Python dependencies required to run the project.
- **project_functions.py**: File listing all functions required to run the project.
- **crime_data.csv**: File which reflects incidents of crime in the City of Los Angeles dating back to 2020 (data.gov)

## Getting Started

To run the code in this project, follow these steps:

1. Ensure you have Python and Jupyter Notebook installed on your system.
2. Clone this repository to your local machine.
3. Install the required Python dependencies using `pip install -r requirements.txt`.
4. Open the Jupyter Notebooks `data_exploration.ipynb` and `data_transformations.ipynb` to explore and run the code.

## Python Packages

The project relies on the following Python packages:

```
from pyspark.sql import SparkSession, DataFrame
import os, logging, re
from pyspark.sql import functions as F
from pyspark.sql.functions import *  
from pyspark.sql.types import * 
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
