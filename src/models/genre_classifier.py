"""
Genre classification functionality for the music genre classifier.
"""

from pyspark.ml import PipelineModel
from pyspark.sql.types import FloatType
import logging
import json
import numpy as np

from src.utils.spark_utils import load_label_mapping, create_lyrics_dataframe
from config.app_config import RF_MODEL_PATH, GBT_MODEL_PATH, LR_MODEL_PATH, LABEL_MAPPING_PATH


class GenreClassifier:
    """Class for making genre predictions on new lyrics."""
    
    def __init__(self, spark):
        """
        Initialize with a SparkSession and load models.
        
        Args:
            spark (SparkSession): The active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self._load_models()
        self._load_label_mapping()
    
    def _load_models(self):
        """Load the trained models."""
        self.logger.info("Loading trained models")
        
        try:
            self.rf_model = PipelineModel.load(RF_MODEL_PATH)
            self.gbt_model = PipelineModel.load(GBT_MODEL_PATH)
            self.lr_model = PipelineModel.load(LR_MODEL_PATH)
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_label_mapping(self):
        """Load the label mapping."""
        self.logger.info("Loading label mapping")
        
        try:
            self.label_mapping = load_label_mapping(self.spark, LABEL_MAPPING_PATH)
            self.logger.info("Label mapping loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading label mapping: {str(e)}")
            raise
    
    def classify_lyrics(self, lyrics_text, year=2025):
        """
        Classify new lyrics using the ensemble of trained models.
        
        Args:
            lyrics_text (str): The lyrics to classify
            year (int, optional): The year to associate with the lyrics. Defaults to current year.
            
        Returns:
            dict: Classification results with ensemble prediction and model details
        """
        self.logger.info("Classifying new lyrics")
        
        try:
            # Create a DataFrame with the new lyrics
            lyrics_df = create_lyrics_dataframe(self.spark, lyrics_text, year)
            
            # Get predictions from all models
            rf_pred = self.rf_model.transform(lyrics_df)
            gbt_pred = self.gbt_model.transform(lyrics_df)
            lr_pred = self.lr_model.transform(lyrics_df)
            
            # Extract prediction and probability from each model
            rf_result = rf_pred.select("prediction", "probability").collect()[0]
            rf_label = rf_result.prediction
            rf_probs = rf_result.probability.toArray()
            
            gbt_result = gbt_pred.select("prediction", "probability").collect()[0]
            gbt_label = gbt_result.prediction
            gbt_probs = gbt_result.probability.toArray()
            
            lr_result = lr_pred.select("prediction", "probability").collect()[0]
            lr_label = lr_result.prediction
            lr_probs = lr_result.probability.toArray()
            
            # Majority voting for ensemble prediction
            votes = {}
            for pred in [rf_label, gbt_label, lr_label]:
                if pred in votes:
                    votes[pred] += 1
                else:
                    votes[pred] = 1
            
            # Find prediction with most votes
            max_votes = 0
            ensemble_label = None
            for pred, vote_count in votes.items():
                if vote_count > max_votes:
                    max_votes = vote_count
                    ensemble_label = pred
            
            # In case of tie, use Random Forest prediction
            if ensemble_label is None:
                ensemble_label = rf_label
            
            # Map numeric labels to genre names
            rf_genre = self.label_mapping[float(rf_label)]
            gbt_genre = self.label_mapping[float(gbt_label)]
            lr_genre = self.label_mapping[float(lr_label)]
            ensemble_genre = self.label_mapping[float(ensemble_label)]
            
            # Create probability dictionaries for each model
            rf_genre_probs = {self.label_mapping[float(i)]: float(rf_probs[i]) 
                             for i in range(len(rf_probs))}
            gbt_genre_probs = {self.label_mapping[float(i)]: float(gbt_probs[i]) 
                              for i in range(len(gbt_probs))}
            lr_genre_probs = {self.label_mapping[float(i)]: float(lr_probs[i]) 
                             for i in range(len(lr_probs))}
            
            # Calculate weighted ensemble probabilities
            rf_weight = 0.40
            gbt_weight = 0.35
            lr_weight = 0.25
            
            ensemble_probs = {}
            for i in range(len(rf_probs)):
                genre = self.label_mapping[float(i)]
                weighted_prob = (rf_probs[i] * rf_weight) + \
                                (gbt_probs[i] * gbt_weight) + \
                                (lr_probs[i] * lr_weight)
                ensemble_probs[genre] = float(weighted_prob)
            
            # Normalize ensemble probabilities
            total = sum(ensemble_probs.values())
            for genre in ensemble_probs:
                ensemble_probs[genre] /= total
            
            # Create comprehensive result with all model predictions
            result = {
                "ensemble_prediction": ensemble_genre,
                "ensemble_probabilities": ensemble_probs,
                "model_predictions": {
                    "random_forest": {
                        "prediction": rf_genre,
                        "probabilities": rf_genre_probs
                    },
                    "gradient_boosting": {
                        "prediction": gbt_genre,
                        "probabilities": gbt_genre_probs
                    },
                    "logistic_regression": {
                        "prediction": lr_genre,
                        "probabilities": lr_genre_probs
                    }
                }
            }
            
            self.logger.info(f"Classification completed. Ensemble prediction: {ensemble_genre}")
            return result
        except Exception as e:
            self.logger.error(f"Error classifying lyrics: {str(e)}")
            raise


"""
Module for the lyrics classifier used for real-time classification.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StringType, StructType, StructField, FloatType


'''class LyricsClassifier:
    """Class for classifying lyrics using trained models."""
    
    def __init__(self, spark, rf_model, gbt_model, lr_model):
        """
        Initialize the classifier with trained models.
        
        Args:
            spark: SparkSession
            rf_model: Trained RandomForest model
            gbt_model: Trained GradientBoosting model
            lr_model: Trained LogisticRegression model
        """
        self.spark = spark
        self.rf_model = rf_model
        self.gbt_model = gbt_model
        self.lr_model = lr_model
        
        # Get label mapping from the indexer
        self.label_mapping = {float(idx): genre for idx, genre in enumerate(rf_model.stages[0].labels)}
        
        # Create a UDF for text length feature
        self.text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
    
    def _create_dataframe_from_lyrics(self, lyrics_text):
        """
        Create a DataFrame with the new lyrics.
        
        Args:
            lyrics_text: String containing lyrics to classify
            
        Returns:
            DataFrame with lyrics and required features
        """
        schema = StructType([
            StructField("artist_name", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("genre", StringType(), True),
            StructField("lyrics", StringType(), True),
            StructField("year", IntegerType(), True),
        ])
        
        current_year = 2025  # Default to current year if not specified
        
        new_data = self.spark.createDataFrame(
            [(None, None, str(current_year), "unknown", lyrics_text, current_year)],
            schema=schema
        )
        
        # Add text length feature
        new_data = new_data.withColumn("text_length", self.text_length_udf(col("lyrics")))
        
        return new_data
    
    def classify_lyrics(self, lyrics_text):
        """
        Classify new lyrics using the ensemble of trained models.
        
        Args:
            lyrics_text: String containing lyrics to classify
            
        Returns:
            Dictionary with predicted genre and probabilities from all models
        """
        # Create a DataFrame with the new lyrics
        new_data = self._create_dataframe_from_lyrics(lyrics_text)
        
        # Get predictions from all models
        rf_pred = self.rf_model.transform(new_data)
        gbt_pred = self.gbt_model.transform(new_data)
        lr_pred = self.lr_model.transform(new_data)
        
        # Extract prediction and probability from each model
        rf_result = rf_pred.select("prediction", "probability").collect()[0]
        rf_label = rf_result.prediction
        rf_probs = rf_result.probability.toArray()
        
        gbt_result = gbt_pred.select("prediction", "probability").collect()[0]
        gbt_label = gbt_result.prediction
        gbt_probs = gbt_result.probability.toArray()
        
        lr_result = lr_pred.select("prediction", "probability").collect()[0]
        lr_label = lr_result.prediction
        lr_probs = lr_result.probability.toArray()
        
        # Determine ensemble prediction using majority voting
        votes = {}
        for pred in [rf_label, gbt_label, lr_label]:
            if pred in votes:
                votes[pred] += 1
            else:
                votes[pred] = 1
        
        # Find the prediction with the most votes
        max_votes = 0
        ensemble_label = None
        for pred, vote_count in votes.items():
            if vote_count > max_votes:
                max_votes = vote_count
                ensemble_label = pred
        
        # In case of a tie, use the RandomForest prediction
        if ensemble_label is None:
            ensemble_label = rf_label
        
        # Map predictions to genre names
        rf_genre = self.label_mapping[rf_label]
        gbt_genre = self.label_mapping[gbt_label]
        lr_genre = self.label_mapping[lr_label]
        ensemble_genre = self.label_mapping[float(ensemble_label)]
        
        # Create dictionaries of probabilities for each model
        rf_genre_probs = {self.label_mapping[float(i)]: float(rf_probs[i]) for i in range(len(rf_probs))}
        gbt_genre_probs = {self.label_mapping[float(i)]: float(gbt_probs[i]) for i in range(len(gbt_probs))}
        lr_genre_probs = {self.label_mapping[float(i)]: float(lr_probs[i]) for i in range(len(lr_probs))}
        
        # Calculate weighted ensemble probabilities
        rf_weight = 0.40
        gbt_weight = 0.35
        lr_weight = 0.25
        
        ensemble_probs = {}
        for i in range(len(rf_probs)):
            genre = self.label_mapping[float(i)]
            weighted_prob = (rf_probs[i] * rf_weight) + (gbt_probs[i] * gbt_weight) + (lr_probs[i] * lr_weight)
            ensemble_probs[genre] = float(weighted_prob)
        
        # Normalize ensemble probabilities
        total = sum(ensemble_probs.values())
        for genre in ensemble_probs:
            ensemble_probs[genre] /= total
        
        # Create a comprehensive result with all model predictions
        result = {
            "ensemble_prediction": ensemble_genre,
            "ensemble_probabilities": ensemble_probs,
            "model_predictions": {
                "random_forest": {
                    "prediction": rf_genre,
                    "probabilities": rf_genre_probs
                },
                "gradient_boosting": {
                    "prediction": gbt_genre,
                    "probabilities": gbt_genre_probs
                },
                "logistic_regression": {
                    "prediction": lr_genre,
                    "probabilities": lr_genre_probs
                }
            }
        }
        
        return result
'''
class LyricsClassifier:
    """Class for classifying lyrics using trained models."""
    
    def __init__(self, rf_model, gbt_model, lr_model, text_length_udf=None):
        """
        Initialize with trained models.
        
        Args:
            rf_model: Trained Random Forest model
            gbt_model: Trained Gradient Boosted Trees model
            lr_model: Trained Logistic Regression model
            text_length_udf: User-defined function for calculating text length
        """
        self.rf_model = rf_model
        self.gbt_model = gbt_model
        self.lr_model = lr_model
        self.text_length_udf = text_length_udf
        self.logger = logging.getLogger(__name__)
        
        # Get the model labels from the RandomForest model
        self.model_labels = self.rf_model.stages[0].labels
        
    def classify_lyrics(self, lyrics_text, year=None):
        """
        Classify lyrics using the ensemble of trained models.
        
        Args:
            lyrics_text (str): The lyrics to classify
            year (int, optional): The release year. Defaults to current year.
            
        Returns:
            dict: Ensemble prediction and individual model predictions
        """
        from pyspark.sql.types import StringType, IntegerType, StructType, StructField
        from pyspark.sql.functions import col
        from pyspark.sql import SparkSession
        import numpy as np
        
        # Get current Spark session
        spark = SparkSession.builder.getOrCreate()
        
        try:
            # Use current year if not specified
            current_year = year if year else 2025
            
            # Create a DataFrame with the new lyrics
            schema = StructType([
                StructField("artist_name", StringType(), True),
                StructField("track_name", StringType(), True),
                StructField("release_date", StringType(), True),
                StructField("genre", StringType(), True),
                StructField("lyrics", StringType(), True),
                StructField("year", IntegerType(), True),
            ])
            
            new_data = spark.createDataFrame(
                [(None, None, str(current_year), "unknown", lyrics_text, current_year)],
                schema=schema
            )
            
            # Add text length feature if UDF is provided
            if self.text_length_udf:
                new_data = new_data.withColumn("text_length", self.text_length_udf(col("lyrics")))
            
            # Get predictions from all models
            rf_pred = self.rf_model.transform(new_data)
            gbt_pred = self.gbt_model.transform(new_data)
            lr_pred = self.lr_model.transform(new_data)
            
            # Extract prediction and probability from each model
            rf_result = rf_pred.select("prediction", "probability").collect()[0]
            rf_label = rf_result.prediction
            rf_probs = rf_result.probability.toArray()
            
            # For GBT model which may not have probability column
            gbt_result = gbt_pred.select("prediction").collect()[0]
            gbt_label = gbt_result.prediction
            # Create a default probability array with 1.0 for the predicted class
            num_classes = len(self.model_labels)
            gbt_probs = np.zeros(num_classes)
            gbt_probs[int(gbt_label)] = 1.0
            
            lr_result = lr_pred.select("prediction", "probability").collect()[0]
            lr_label = lr_result.prediction
            lr_probs = lr_result.probability.toArray()
            
            # Determine ensemble prediction using majority voting
            votes = {}
            for pred in [rf_label, gbt_label, lr_label]:
                if pred in votes:
                    votes[pred] += 1
                else:
                    votes[pred] = 1
            
            # Find prediction with most votes
            max_votes = 0
            ensemble_label = None
            for pred, vote_count in votes.items():
                if vote_count > max_votes:
                    max_votes = vote_count
                    ensemble_label = pred
            
            # In case of tie, use RandomForest prediction
            if ensemble_label is None:
                ensemble_label = rf_label
            
            # Map predictions to genre names
            label_mapping = {float(idx): genre for idx, genre in enumerate(self.model_labels)}
            
            rf_genre = label_mapping[rf_label]
            gbt_genre = label_mapping[gbt_label]
            lr_genre = label_mapping[lr_label]
            ensemble_genre = label_mapping[float(ensemble_label)]
            
            # Create dictionaries of probabilities for each model
            rf_genre_probs = {label_mapping[float(i)]: float(rf_probs[i]) 
                             for i in range(len(rf_probs)) if float(i) in label_mapping}
            gbt_genre_probs = {label_mapping[float(i)]: float(gbt_probs[i]) 
                             for i in range(len(gbt_probs)) if float(i) in label_mapping}
            lr_genre_probs = {label_mapping[float(i)]: float(lr_probs[i]) 
                             for i in range(len(lr_probs)) if float(i) in label_mapping}
            
            # Calculate weighted ensemble probabilities
            rf_weight = 0.40
            gbt_weight = 0.35
            lr_weight = 0.25
            
            ensemble_probs = {}
            for i in range(len(self.model_labels)):
                genre = label_mapping[float(i)]
                weighted_prob = (rf_probs[i] * rf_weight) + (gbt_probs[i] * gbt_weight) + (lr_probs[i] * lr_weight)
                ensemble_probs[genre] = float(weighted_prob)
            
            # Normalize ensemble probabilities
            total = sum(ensemble_probs.values())
            for genre in ensemble_probs:
                ensemble_probs[genre] /= total
            
            # Create comprehensive result
            result = {
                "ensemble_prediction": ensemble_genre,
                "ensemble_probabilities": ensemble_probs,
                "model_predictions": {
                    "random_forest": {
                        "prediction": rf_genre,
                        "probabilities": rf_genre_probs
                    },
                    "gradient_boosting": {
                        "prediction": gbt_genre,
                        "probabilities": gbt_genre_probs
                    },
                    "logistic_regression": {
                        "prediction": lr_genre,
                        "probabilities": lr_genre_probs
                    }
                }
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Error classifying lyrics: {str(e)}")
            raise
    
