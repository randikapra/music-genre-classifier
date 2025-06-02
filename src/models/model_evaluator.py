"""
Model evaluation functionality for the music genre classifier.
"""

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, udf, avg
from pyspark.sql.types import FloatType, DoubleType
import logging
from pyspark.sql.functions import col


class ModelEvaluator:
    """Class for evaluating trained models."""
    
    def __init__(self, spark):
        """
        Initialize with a SparkSession.
        
        Args:
            spark (SparkSession): The active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model, test_df, model_name):
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            test_df (DataFrame): Test dataset
            model_name (str): Name of the model for logging
            
        Returns:
            DataFrame: Predictions DataFrame
        """
        self.logger.info(f"Evaluating {model_name} model")
        
        try:
            # Generate predictions
            predictions = model.transform(test_df)
            
            # Create evaluators for different metrics
            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy")
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="f1")
            evaluator_precision = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
            evaluator_recall = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="weightedRecall")
            
            # Calculate metrics
            accuracy = evaluator_accuracy.evaluate(predictions)
            f1 = evaluator_f1.evaluate(predictions)
            precision = evaluator_precision.evaluate(predictions)
            recall = evaluator_recall.evaluate(predictions)
            
            # Log results
            self.logger.info(f"{model_name} Test Metrics:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"F1 Score: {f1:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            
            return predictions
        except Exception as e:
            self.logger.error(f"Error evaluating {model_name} model: {str(e)}")
            raise
    
    def create_majority_vote_ensemble(self, rf_preds, gbt_preds, lr_preds):
        """
        Create ensemble predictions using majority voting.
        
        Args:
            rf_preds (DataFrame): Random Forest predictions
            gbt_preds (DataFrame): GBT predictions
            lr_preds (DataFrame): Logistic Regression predictions
            
        Returns:
            DataFrame: Ensemble predictions
        """
        self.logger.info("Creating ensemble predictions using majority voting")
        
        try:
            # Extract predictions from each model
            rf_data = rf_preds.select("artist_name", "track_name", "genre", "label", 
                                     rf_preds["prediction"].alias("rf_prediction"))
            
            gbt_data = gbt_preds.select("artist_name", "track_name", "genre", "label",
                                       gbt_preds["prediction"].alias("gbt_prediction"))
            
            lr_data = lr_preds.select("artist_name", "track_name", "genre", "label",
                                     lr_preds["prediction"].alias("lr_prediction"))
            
            # Join the dataframes
            joined_df = rf_data.join(gbt_data, ["artist_name", "track_name", "genre", "label"])
            joined_df = joined_df.join(lr_data, ["artist_name", "track_name", "genre", "label"])
            
            # Create majority vote UDF
            def majority_vote(rf_pred, gbt_pred, lr_pred):
                votes = {}
                for pred in [rf_pred, gbt_pred, lr_pred]:
                    if pred in votes:
                        votes[pred] += 1
                    else:
                        votes[pred] = 1
                
                # Find prediction with most votes
                max_votes = 0
                max_pred = None
                for pred, vote_count in votes.items():
                    if vote_count > max_votes:
                        max_votes = vote_count
                        max_pred = pred
                
                # In case of tie, use Random Forest prediction
                return float(max_pred if max_pred is not None else rf_pred)
            
            # Register the UDF
            majority_vote_udf = udf(majority_vote, FloatType())
            
            # Apply the UDF
            ensemble_df = joined_df.withColumn(
                "prediction",
                majority_vote_udf("rf_prediction", "gbt_prediction", "lr_prediction")
            )
            
            # Add indicators for which models were correct
            ensemble_df = ensemble_df.withColumn("rf_correct", 
                                                when(col("rf_prediction") == col("label"), 1).otherwise(0))
            ensemble_df = ensemble_df.withColumn("gbt_correct", 
                                                when(col("gbt_prediction") == col("label"), 1).otherwise(0))
            ensemble_df = ensemble_df.withColumn("lr_correct", 
                                                when(col("lr_prediction") == col("label"), 1).otherwise(0))
            ensemble_df = ensemble_df.withColumn("ensemble_correct", 
                                                when(col("prediction") == col("label"), 1).otherwise(0))
            
            return ensemble_df
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {str(e)}")
            raise
    
    def evaluate_ensemble(self, ensemble_preds):
        """
        Evaluate the ensemble model.
        
        Args:
            ensemble_preds (DataFrame): Ensemble predictions
            
        Returns:
            dict: Dictionary with ensemble evaluation metrics
        """
        self.logger.info("Evaluating ensemble model")
        
        try:
            # Create evaluators
            evaluator_accuracy = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy")
            evaluator_f1 = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="f1")
            evaluator_precision = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
            evaluator_recall = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="weightedRecall")
            
            # Ensure prediction column is of type DoubleType
            ensemble_preds = ensemble_preds.withColumn("prediction", col("prediction").cast(DoubleType()))

            # Calculate metrics
            ensemble_accuracy = evaluator_accuracy.evaluate(ensemble_preds)
            ensemble_f1 = evaluator_f1.evaluate(ensemble_preds)
            ensemble_precision = evaluator_precision.evaluate(ensemble_preds)
            ensemble_recall = evaluator_recall.evaluate(ensemble_preds)
            
            # Log results
            self.logger.info(f"Ensemble Model Test Metrics:")
            self.logger.info(f"Accuracy: {ensemble_accuracy:.4f}")
            self.logger.info(f"F1 Score: {ensemble_f1:.4f}")
            self.logger.info(f"Precision: {ensemble_precision:.4f}")
            self.logger.info(f"Recall: {ensemble_recall:.4f}")
            
            # Compare model performance
            model_correct_counts = ensemble_preds.agg(
                avg("rf_correct").alias("RandomForest"),
                avg("gbt_correct").alias("GradientBoosting"),
                avg("lr_correct").alias("LogisticRegression"),
                avg("ensemble_correct").alias("Ensemble")
            ).collect()[0]
            
            self.logger.info("Model Performance Comparison:")
            self.logger.info(f"RandomForest Accuracy: {model_correct_counts['RandomForest']:.4f}")
            self.logger.info(f"GradientBoosting Accuracy: {model_correct_counts['GradientBoosting']:.4f}")
            self.logger.info(f"LogisticRegression Accuracy: {model_correct_counts['LogisticRegression']:.4f}")
            self.logger.info(f"Ensemble Accuracy: {model_correct_counts['Ensemble']:.4f}")
            
            # Create confusion matrix
            self.logger.info("Generating confusion matrix")
            confusion_matrix = ensemble_preds.groupBy("label", "prediction").count().orderBy("label", "prediction")
            
            return {
                "accuracy": ensemble_accuracy,
                "f1": ensemble_f1,
                "precision": ensemble_precision,
                "recall": ensemble_recall,
                "model_comparison": model_correct_counts,
                "confusion_matrix": confusion_matrix
            }
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble: {str(e)}")
            raise

    def compare_models(self, ensemble_preds):
        """
        Compare the performance of different models.
        
        Args:
            ensemble_preds (DataFrame): DataFrame with predictions from all models
        """
        self.logger.info("Comparing model performance...")
        
        try:
            # Get average accuracy for each model
            model_correct_counts = ensemble_preds.agg(
                avg("rf_correct").alias("RandomForest"),
                avg("gbt_correct").alias("GradientBoosting"),
                avg("lr_correct").alias("LogisticRegression"),
                avg("ensemble_correct").alias("Ensemble")
            ).collect()[0]
            
            # Log the comparison results
            print("\nModel Performance Comparison:")
            print(f"RandomForest Accuracy: {model_correct_counts['RandomForest']:.4f}")
            print(f"GradientBoosting Accuracy: {model_correct_counts['GradientBoosting']:.4f}")
            print(f"LogisticRegression Accuracy: {model_correct_counts['LogisticRegression']:.4f}")
            print(f"Ensemble Accuracy: {model_correct_counts['Ensemble']:.4f}")
            
            return model_correct_counts
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise

    def create_classifier(self, rf_model, gbt_model, lr_model):
        """
        Create a lyrics classifier with the trained models.
        
        Args:
            rf_model: Trained Random Forest model
            gbt_model: Trained Gradient Boosted Trees model
            lr_model: Trained Logistic Regression model
            
        Returns:
            LyricsClassifier: A classifier for new lyrics
        """
        from pyspark.sql.functions import udf
        from pyspark.sql.types import IntegerType
        
        self.logger.info("Creating lyrics classifier")
        
        # Create text length UDF
        text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
        
        try:
            # Import locally to avoid circular imports
            from src.models.genre_classifier import LyricsClassifier
            
            # Create and return classifier
            return LyricsClassifier(rf_model, gbt_model, lr_model, text_length_udf)
        except Exception as e:
            self.logger.error(f"Error creating classifier: {str(e)}")
            raise