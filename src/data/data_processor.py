## --version 1.0-- ##
'''
"""
Data processing utilities for the music genre classifier.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import logging

from config.app_config import TRAIN_TEST_SPLIT, RANDOM_SEED, MIN_SAMPLES, MAX_SAMPLES


class DataProcessor:
    """Class for handling data loading, cleaning, and preparation."""
    
    def __init__(self, spark):
        """
        Initialize with a SparkSession.
        
        Args:
            spark (SparkSession): The active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path):
        """
        Load the dataset from CSV.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            DataFrame: Loaded DataFrame
        """
        self.logger.info(f"Loading dataset from {file_path}")
        try:
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            self.logger.info(f"Dataset loaded successfully with {df.count()} rows")
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def clean_data(self, df):
        """
        Clean the dataset by selecting relevant columns and dropping nulls.
        
        Args:
            df (DataFrame): Raw DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        try:
            # Select relevant columns and drop rows with null values
            music_df = df.select('artist_name', 'track_name', 'release_date', 'genre', 'lyrics').na.drop()
            
            # Convert release_date to numeric year
            music_df = music_df.withColumn("year", col("release_date").cast(IntegerType()))
            
            self.logger.info(f"Data cleaned, resulting in {music_df.count()} rows")
            return music_df
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def balance_dataset(self, df):
        """
        Balance the dataset by undersampling majority classes and 
        oversampling minority classes.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            DataFrame: Balanced DataFrame
        """
        self.logger.info("Balancing dataset...")
        
        try:
            # Get genre distribution
            genre_counts = df.groupBy('genre').count().orderBy('count', ascending=False)
            
            # Get minimum count (for undersampling reference)
            min_count = genre_counts.orderBy('count').first()['count']
            
            # Calculate target sample size between min and max thresholds
            target_samples = (MIN_SAMPLES + MAX_SAMPLES) // 2
            
            # Get list of all genres
            genre_list = [row['genre'] for row in genre_counts.select('genre').collect()]
            
            # Process each genre
            balanced_dfs = []
            for genre in genre_list:
                genre_df = df.filter(col('genre') == genre)
                genre_count = genre_df.count()
                
                if genre_count < target_samples:
                    # Oversample minority classes
                    fraction = float(target_samples) / genre_count
                    sampled_df = genre_df.sample(withReplacement=True, fraction=fraction, seed=RANDOM_SEED)
                    self.logger.info(f"Genre '{genre}': {genre_count} → {sampled_df.count()} samples (oversampled)")
                else:
                    # Undersample majority classes
                    fraction = float(target_samples) / genre_count
                    sampled_df = genre_df.sample(withReplacement=False, fraction=fraction, seed=RANDOM_SEED)
                    self.logger.info(f"Genre '{genre}': {genre_count} → {sampled_df.count()} samples (undersampled)")
                
                balanced_dfs.append(sampled_df)
            
            # Combine all balanced dataframes
            balanced_df = balanced_dfs[0]
            for i in range(1, len(balanced_dfs)):
                balanced_df = balanced_df.union(balanced_dfs[i])
            
            # Add text length feature
            text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
            balanced_df = balanced_df.withColumn("text_length", text_length_udf(col("lyrics")))
            
            self.logger.info(f"Balanced dataset created with {balanced_df.count()} rows")
            return balanced_df
        except Exception as e:
            self.logger.error(f"Error balancing dataset: {str(e)}")
            raise
    
    def split_data(self, df):
        """
        Split the data into training and testing sets.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            tuple: (train_df, test_df)
        """
        self.logger.info(f"Splitting data with ratio {TRAIN_TEST_SPLIT}")
        try:
            train_df, test_df = df.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
            # Save the datasets
            train_df.to_csv('/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/data/train_data.csv', index=False)
            test_df.to_csv('/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/data/test_data.csv', index=False)
            print("Data split and saved to train_data.csv and test_data.csv")

            self.logger.info(f"Training dataset size: {train_df.count()}")
            self.logger.info(f"Testing dataset size: {test_df.count()}")
            return train_df, test_df
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

'''


## --version 2.0-- ##
"""
Data processing utilities for the music genre classifier.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
import logging

from config.app_config import TRAIN_TEST_SPLIT, RANDOM_SEED, MIN_SAMPLES, MAX_SAMPLES


class DataProcessor:
    """Class for handling data loading, cleaning, and preparation."""
    
    def __init__(self, spark):
        """
        Initialize with a SparkSession.
        
        Args:
            spark (SparkSession): The active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path):
        """
        Load the dataset from CSV.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            DataFrame: Loaded DataFrame
        """
        self.logger.info(f"Loading dataset from {file_path}")
        try:
            df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            self.logger.info(f"Dataset loaded successfully with {df.count()} rows")
            return df
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def clean_data(self, df):
        """
        Clean the dataset by selecting relevant columns and dropping nulls.
        
        Args:
            df (DataFrame): Raw DataFrame
            
        Returns:
            DataFrame: Cleaned DataFrame
        """
        self.logger.info("Cleaning data...")
        try:
            # Select relevant columns and drop rows with null values
            music_df = df.select('artist_name', 'track_name', 'release_date', 'genre', 'lyrics').na.drop()
            
            # Convert release_date to numeric year
            music_df = music_df.withColumn("year", col("release_date").cast(IntegerType()))
            
            self.logger.info(f"Data cleaned, resulting in {music_df.count()} rows")
            return music_df
        except Exception as e:
            self.logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def balance_dataset(self, df):
        """
        Balance the dataset by undersampling majority classes and 
        oversampling minority classes.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            DataFrame: Balanced DataFrame
        """
        self.logger.info("Balancing dataset...")
        
        try:
            # Get genre distribution
            genre_counts = df.groupBy('genre').count().orderBy('count', ascending=False)
            
            # Get minimum count (for undersampling reference)
            min_count = genre_counts.orderBy('count').first()['count']
            
            # Calculate target sample size between min and max thresholds
            target_samples = (MIN_SAMPLES + MAX_SAMPLES) // 2
            
            # Get list of all genres
            genre_list = [row['genre'] for row in genre_counts.select('genre').collect()]
            
            # Process each genre
            balanced_dfs = []
            for genre in genre_list:
                genre_df = df.filter(col('genre') == genre)
                genre_count = genre_df.count()
                
                if genre_count < target_samples:
                    # Oversample minority classes
                    fraction = float(target_samples) / genre_count
                    sampled_df = genre_df.sample(withReplacement=True, fraction=fraction, seed=RANDOM_SEED)
                    self.logger.info(f"Genre '{genre}': {genre_count} → {sampled_df.count()} samples (oversampled)")
                else:
                    # Undersample majority classes
                    fraction = float(target_samples) / genre_count
                    sampled_df = genre_df.sample(withReplacement=False, fraction=fraction, seed=RANDOM_SEED)
                    self.logger.info(f"Genre '{genre}': {genre_count} → {sampled_df.count()} samples (undersampled)")
                
                balanced_dfs.append(sampled_df)
            
            # Combine all balanced dataframes
            balanced_df = balanced_dfs[0]
            for i in range(1, len(balanced_dfs)):
                balanced_df = balanced_df.union(balanced_dfs[i])
            
            # Add text length feature
            text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
            balanced_df = balanced_df.withColumn("text_length", text_length_udf(col("lyrics")))
            
            self.logger.info(f"Balanced dataset created with {balanced_df.count()} rows")
            return balanced_df
        except Exception as e:
            self.logger.error(f"Error balancing dataset: {str(e)}")
            raise
    
    def split_data(self, df):
        """
        Split the data into training and testing sets.
        
        Args:
            df (DataFrame): Input DataFrame
            
        Returns:
            tuple: (train_df, test_df)
        """
        self.logger.info(f"Splitting data with ratio {TRAIN_TEST_SPLIT}")
        try:
            train_df, test_df = df.randomSplit(TRAIN_TEST_SPLIT, seed=RANDOM_SEED)
            
            # # Save the datasets using PySpark's write.csv() method
            # train_path = '/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/data/train_data.csv'
            # test_path = '/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/data/test_data.csv'
            
            # # Write train data
            # train_df.write.mode("overwrite").option("header", "true").csv(train_path)
            # self.logger.info(f"Training dataset saved to {train_path}")
            
            # # Write test data
            # test_df.write.mode("overwrite").option("header", "true").csv(test_path)
            # self.logger.info(f"Testing dataset saved to {test_path}")

            self.logger.info(f"Training dataset size: {train_df.count()}")
            self.logger.info(f"Testing dataset size: {test_df.count()}")
            return train_df, test_df
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise