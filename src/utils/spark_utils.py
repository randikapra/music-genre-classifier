"""
Spark utilities for the music genre classifier.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import logging

from config.app_config import SPARK_CONFIG

def create_spark_session(app_name=None, driver_memory=None, executor_memory=None):
    """
    Create and configure a Spark session.
    
    Args:
        app_name (str, optional): Name of the Spark application
        driver_memory (str, optional): Amount of memory to allocate to the driver
        executor_memory (str, optional): Amount of memory to allocate to each executor
    
    Returns:
        SparkSession: Configured Spark session
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Spark session")
    
    try:
        # Use provided parameters or fall back to config values
        app_name = app_name or SPARK_CONFIG["app_name"]
        driver_memory = driver_memory or SPARK_CONFIG["driver_memory"]
        executor_memory = executor_memory or SPARK_CONFIG["executor_memory"]
        
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", driver_memory) \
            .config("spark.executor.memory", executor_memory) \
            .getOrCreate()
        
        logger.info(f"Spark session created successfully with app_name={app_name}, driver_memory={driver_memory}, executor_memory={executor_memory}")
        return spark
    except Exception as e:
        logger.error(f"Error creating Spark session: {str(e)}")
        raise

# def create_spark_session():
#     """
#     Create and configure a Spark session.
    
#     Returns:
#         SparkSession: Configured Spark session
#     """
#     logger = logging.getLogger(__name__)
#     logger.info("Creating Spark session")
    
#     try:
#         spark = SparkSession.builder \
#             .appName(SPARK_CONFIG["app_name"]) \
#             .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
#             .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
#             .getOrCreate()
        
#         logger.info("Spark session created successfully")
#         return spark
#     except Exception as e:
#         logger.error(f"Error creating Spark session: {str(e)}")
#         raise


def load_label_mapping(spark, label_mapping_path):
    """
    Load label mapping from CSV.
    
    Args:
        spark (SparkSession): Active Spark session
        label_mapping_path (str): Path to label mapping CSV
        
    Returns:
        dict: Label ID to genre name mapping
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading label mapping from {label_mapping_path}")
    
    try:
        # Read the CSV file with label mapping
        label_df = spark.read.csv(label_mapping_path, header=True, inferSchema=True)
        
        # Convert to dictionary
        label_mapping = {row['label_id']: row['genre'] for row in label_df.collect()}
        
        logger.info(f"Label mapping loaded successfully with {len(label_mapping)} labels")
        return label_mapping
    except Exception as e:
        logger.error(f"Error loading label mapping: {str(e)}")
        raise


def create_lyrics_dataframe(spark, lyrics_text, year=2025):
    """
    Create a DataFrame with new lyrics for prediction.
    
    Args:
        spark (SparkSession): Active Spark session
        lyrics_text (str): The lyrics to classify
        year (int, optional): The year to associate with the lyrics. Defaults to current year.
        
    Returns:
        DataFrame: DataFrame containing the lyrics for prediction
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating DataFrame for new lyrics")
    
    try:
        # Define schema similar to the training data
        schema = StructType([
            StructField("artist_name", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("genre", StringType(), True),
            StructField("lyrics", StringType(), True),
            StructField("year", IntegerType(), True),
        ])
        
        # Create DataFrame with the new lyrics
        lyrics_df = spark.createDataFrame(
            [(None, None, str(year), "unknown", lyrics_text, year)],
            schema=schema
        )
        
        # Add text length feature
        text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
        lyrics_df = lyrics_df.withColumn("text_length", text_length_udf(col("lyrics")))
        
        logger.info("Lyrics DataFrame created successfully")
        return lyrics_df
    except Exception as e:
        logger.error(f"Error creating lyrics DataFrame: {str(e)}")
        raise