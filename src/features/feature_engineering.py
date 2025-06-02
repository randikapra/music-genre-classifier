"""
Feature engineering for the music genre classifier.
"""

### --version 1.0-- ##
'''
from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover, 
                               CountVectorizer, IDF, StringIndexer, VectorAssembler)
import logging

from config.app_config import MUSIC_STOP_WORDS


class FeatureEngineering:
    """Class for creating feature engineering pipeline stages."""
    
    def __init__(self):
        """Initialize the feature engineering class."""
        self.logger = logging.getLogger(__name__)
    
    def create_label_indexer(self):
        """
        Create label indexer for target conversion.
        
        Returns:
            StringIndexer: The configured label indexer
        """
        self.logger.info("Creating label indexer")
        return StringIndexer(
            inputCol="genre", 
            outputCol="label", 
            handleInvalid="keep"
        )
    
    def create_tokenizer(self):
        """
        Create tokenizer for lyrics text.
        
        Returns:
            RegexTokenizer: The configured tokenizer
        """
        self.logger.info("Creating tokenizer")
        return RegexTokenizer(
            inputCol="lyrics", 
            outputCol="words", 
            pattern="[\\s,.!?;:'\"\(\)\[\]\\\/\-_]", 
            toLowercase=True
        )
    
    def create_stopwords_remover(self):
        """
        Create stop words remover with music-specific stop words.
        
        Returns:
            StopWordsRemover: The configured stop words remover
        """
        self.logger.info("Creating stop words remover")
        # Combine standard stop words with music-specific ones
        custom_stop_words = StopWordsRemover.loadDefaultStopWords("english") + MUSIC_STOP_WORDS
        
        return StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words", 
            stopWords=custom_stop_words
        )
    
    def create_count_vectorizer(self):
        """
        Create count vectorizer for word frequency encoding.
        
        Returns:
            CountVectorizer: The configured count vectorizer
        """
        self.logger.info("Creating count vectorizer")
        return CountVectorizer(
            inputCol="filtered_words", 
            outputCol="raw_features",
            minDF=3.0,  # Minimum document frequency
            maxDF=0.8   # Maximum document frequency (as fraction)
        )
    
    def create_idf(self):
        """
        Create IDF transformer for TF-IDF calculation.
        
        Returns:
            IDF: The configured IDF transformer
        """
        self.logger.info("Creating IDF transformer")
        return IDF(
            inputCol="raw_features", 
            outputCol="text_features", 
            minDocFreq=3
        )
    
    def create_vector_assembler(self):
        """
        Create vector assembler to combine all features.
        
        Returns:
            VectorAssembler: The configured vector assembler
        """
        self.logger.info("Creating vector assembler")
        return VectorAssembler(
            inputCols=["text_features", "text_length", "year"],
            outputCol="features",
            handleInvalid="keep"
        )
    
    def get_feature_pipeline_stages(self):
        """
        Get all feature engineering pipeline stages.
        
        Returns:
            list: List of pipeline stages
        """
        self.logger.info("Creating full feature engineering pipeline")
        return [
            self.create_label_indexer(),
            self.create_tokenizer(),
            self.create_stopwords_remover(),
            self.create_count_vectorizer(),
            self.create_idf(),
            self.create_vector_assembler()
        ]
    
'''

from pyspark.ml.feature import (RegexTokenizer, StopWordsRemover, 
                               CountVectorizer, IDF, StringIndexer, VectorAssembler)
import logging

from config.app_config import MUSIC_STOP_WORDS


class FeatureEngineering:
    """Class for creating feature engineering pipeline stages."""
    
    def __init__(self, spark):
        """
        Initialize the feature engineering class.
        
        Args:
            spark: SparkSession object
        """
        self.logger = logging.getLogger(__name__)
        self.spark = spark
    
    def create_label_indexer(self):
        """
        Create label indexer for target conversion.
        
        Returns:
            StringIndexer: The configured label indexer
        """
        self.logger.info("Creating label indexer")
        return StringIndexer(
            inputCol="genre", 
            outputCol="label", 
            handleInvalid="keep"
        )
    
    def create_tokenizer(self):
        """
        Create tokenizer for lyrics text.
        
        Returns:
            RegexTokenizer: The configured tokenizer
        """
        self.logger.info("Creating tokenizer")
        return RegexTokenizer(
            inputCol="lyrics", 
            outputCol="words", 
            pattern="[\\s,.!?;:'\"\(\)\[\]\\\/\-_]", 
            toLowercase=True
        )
    
    def create_stopwords_remover(self):
        """
        Create stop words remover with music-specific stop words.
        
        Returns:
            StopWordsRemover: The configured stop words remover
        """
        self.logger.info("Creating stop words remover")
        # Combine standard stop words with music-specific ones
        custom_stop_words = StopWordsRemover.loadDefaultStopWords("english") + MUSIC_STOP_WORDS
        
        return StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words", 
            stopWords=custom_stop_words
        )
    
    def create_count_vectorizer(self):
        """
        Create count vectorizer for word frequency encoding.
        
        Returns:
            CountVectorizer: The configured count vectorizer
        """
        self.logger.info("Creating count vectorizer")
        return CountVectorizer(
            inputCol="filtered_words", 
            outputCol="raw_features",
            minDF=3.0,  # Minimum document frequency
            maxDF=0.8   # Maximum document frequency (as fraction)
        )
    
    def create_idf(self):
        """
        Create IDF transformer for TF-IDF calculation.
        
        Returns:
            IDF: The configured IDF transformer
        """
        self.logger.info("Creating IDF transformer")
        return IDF(
            inputCol="raw_features", 
            outputCol="text_features", 
            minDocFreq=3
        )
    
    def create_vector_assembler(self):
        """
        Create vector assembler to combine all features.
        
        Returns:
            VectorAssembler: The configured vector assembler
        """
        self.logger.info("Creating vector assembler")
        return VectorAssembler(
            inputCols=["text_features", "text_length", "year"],
            outputCol="features",
            handleInvalid="keep"
        )
    
    def get_feature_pipeline_stages(self):
        """
        Get all feature engineering pipeline stages.
        
        Returns:
            list: List of pipeline stages
        """
        self.logger.info("Creating full feature engineering pipeline")
        return [
            self.create_label_indexer(),
            self.create_tokenizer(),
            self.create_stopwords_remover(),
            self.create_count_vectorizer(),
            self.create_idf(),
            self.create_vector_assembler()
        ]
    
    def add_text_features(self, df):
        """
        Add text features to the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame: Dataframe with added text features
        """
        self.logger.info("Adding text features to dataframe")
        from pyspark.sql.functions import col, udf
        from pyspark.sql.types import IntegerType
        
        # Extract text length as a feature
        text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
        return df.withColumn("text_length", text_length_udf(col("lyrics")))