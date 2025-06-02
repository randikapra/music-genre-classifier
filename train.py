###########################################################################################################
# ============================================= version 1.0 ============================================= #
###########################################################################################################
'''
"""
Main script for training the music genre classifier models.
"""

import logging
import os
import sys

from src.data.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.spark_utils import create_spark_session
from config.app_config import DATASET_PATH


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting music genre classifier training")
    
    try:
        # Check if dataset exists
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset not found at {DATASET_PATH}")
            logger.info("Please download the Mendeley music dataset and place it in the data directory")
            return
        
        # Create Spark session
        spark = create_spark_session()
        
        # Initialize components
        data_processor = DataProcessor(spark)
        model_trainer = ModelTrainer(spark)
        model_evaluator = ModelEvaluator(spark)
        
        # Load and process data
        logger.info("Loading and processing data")
        raw_df = data_processor.load_data(DATASET_PATH)
        cleaned_df = data_processor.clean_data(raw_df)
        balanced_df = data_processor.balance_dataset(cleaned_df)
        train_df, test_df = data_processor.split_data(balanced_df)
        
        # Train models
        logger.info("Training models")
        rf_model, gbt_model, lr_model = model_trainer.train_models(train_df)
        
        # Save models
        logger.info("Saving models")
        model_trainer.save_models(rf_model, gbt_model, lr_model)
        
        # Evaluate models
        logger.info("Evaluating models")
        rf_preds = model_evaluator.evaluate_model(rf_model, test_df, "RandomForest")
        gbt_preds = model_evaluator.evaluate_model(gbt_model, test_df, "GradientBoostedTrees")
        lr_preds = model_evaluator.evaluate_model(lr_model, test_df, "LogisticRegression")
        
        # Evaluate ensemble
        logger.info("Evaluating ensemble")
        ensemble_preds = model_evaluator.create_majority_vote_ensemble(rf_preds, gbt_preds, lr_preds)
        ensemble_metrics = model_evaluator.evaluate_ensemble(ensemble_preds)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
    
'''


###########################################################################################################
# ============================================= version 2.0 ============================================= #
###########################################################################################################
"""
Main script for training the music genre classifier models.
"""

import logging
import os
import sys
import pandas as pd
from pyspark.sql.functions import col

from src.data.data_processor import DataProcessor
from src.features.feature_engineering import FeatureEngineering
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.utils.spark_utils import create_spark_session
from config.app_config import DATASET_PATH, MODEL_BASE_PATH


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)

'''
def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting music genre classifier training")
    
    try:
        # Check if dataset exists
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset not found at {DATASET_PATH}")
            logger.info("Please download the Mendeley music dataset and place it in the data directory")
            return
        
        # Create Spark session with increased memory like in original script
        spark = create_spark_session()
        
        # Initialize components
        data_processor = DataProcessor(spark)
        feature_engineer = FeatureEngineering(spark)
        model_trainer = ModelTrainer(spark)
        model_evaluator = ModelEvaluator(spark)
        
        # Load data
        logger.info("Loading data...")
        raw_df = data_processor.load_data(DATASET_PATH)
        
        # Display schema and sample data like in original script
        print("Dataset Schema:")
        raw_df.printSchema()
        print("\nSample Data:")
        raw_df.show(5, truncate=False)
        
        # Process and clean data
        logger.info("Cleaning data...")
        cleaned_df = data_processor.clean_data(raw_df)
        
        # Display genre distribution like in original script
        print("\nGenre Distribution:")
        genre_counts = cleaned_df.groupBy('genre').count().orderBy('count', ascending=False)
        genre_counts.show()
        
        # Balance dataset
        logger.info("Balancing dataset...")
        # balanced_df = data_processor.balance_dataset(cleaned_df, genre_counts)
        balanced_df = data_processor.balance_dataset(cleaned_df)
        
        # Show new distribution like in original script
        print("\nBalanced Genre Distribution:")
        balanced_df.groupBy('genre').count().orderBy('count', ascending=False).show()
        
        # Extract text features
        logger.info("Extracting text features...")
        featured_df = feature_engineer.add_text_features(balanced_df)
        
        # Split data
        train_df, test_df = data_processor.split_data(featured_df)
        print(f"\nTraining dataset size: {train_df.count()}")
        print(f"Testing dataset size: {test_df.count()}")
        
        # Train models
        logger.info("Training models...")
        print("\nTraining RandomForest model...")
        rf_model = model_trainer.create_random_forest_pipeline(train_df)
        
        print("\nTraining GBT model...")
        gbt_model = model_trainer.create_gbt_pipeline(train_df)
        
        print("\nTraining LogisticRegression model...")
        lr_model = model_trainer.create_logistic_regression_pipeline(train_df)
        
        # Save models
        logger.info("Saving models...")
        rf_path = f"{MODEL_BASE_PATH}_rf"
        gbt_path = f"{MODEL_BASE_PATH}_gbt"
        lr_path = f"{MODEL_BASE_PATH}_lr"
        
        model_trainer.save_model(rf_model, rf_path)
        model_trainer.save_model(gbt_model, gbt_path)
        model_trainer.save_model(lr_model, lr_path)
        
        # Evaluate individual models
        logger.info("Evaluating models...")
        rf_predictions = model_evaluator.evaluate_model(rf_model, test_df, "RandomForest")
        gbt_predictions = model_evaluator.evaluate_model(gbt_model, test_df, "GradientBoostedTrees")
        lr_predictions = model_evaluator.evaluate_model(lr_model, test_df, "LogisticRegression")
        
        # Get label mapping from the indexer and print
        label_mapping = {float(idx): genre for idx, genre in enumerate(rf_model.stages[0].labels)}
        print("\nLabel mapping:", label_mapping)
        
        # Create ensemble predictions using majority voting
        logger.info("Creating ensemble predictions using majority voting...")
        ensemble_preds = model_evaluator.create_majority_vote_ensemble(
            rf_predictions, gbt_predictions, lr_predictions
        )
        
        # Evaluate ensemble model
        model_evaluator.evaluate_ensemble(ensemble_preds)
        
        # Compare model performance
        logger.info("Comparing model performance...")
        model_evaluator.compare_models(ensemble_preds)
        
        # Display confusion matrix for the ensemble model
        print("\nEnsemble Model Confusion Matrix:")
        confusion_matrix = ensemble_preds.groupBy("label", "prediction").count().orderBy("label", "prediction")
        confusion_matrix.show()
        
        # Test the classifier with examples (like in original script)
        print("\n--- Testing the ensemble classifier with example lyrics ---")
        examples = [
            "I feel the rhythm in my soul as I dance through the night with stars above",
            "Broken hearts and whiskey, driving down these country roads thinking of you",
            "The beat drops hard, I'm spitting rhymes in the cipher, mic check one two",
            "Swinging notes and saxophone, the band plays jazz until dawn breaks through",
            "I got the blues in my heart, guitar crying with me all night long",
            "Rock and roll all night, guitar solos blazing as the crowd goes wild",
            "Jamming to the reggae beat, island vibes and sunshine feeling alright"
        ]
        
        classifier = model_evaluator.create_classifier(rf_model, gbt_model, lr_model)
        for example in examples:
            result = classifier.classify_lyrics(example)
            
            print(f"\nExample lyrics: '{example}'")
            print(f"Ensemble prediction: {result['ensemble_prediction']}")
            
            print("\nTop 3 genres by ensemble probability:")
            i = 0
            for genre, prob in sorted(result['ensemble_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"{genre}: {prob:.4f}")
                i += 1
                if i >= 3:
                    break
            
            print("\nIndividual model predictions:")
            print(f"RandomForest: {result['model_predictions']['random_forest']['prediction']}")
            print(f"GradientBoosting: {result['model_predictions']['gradient_boosting']['prediction']}")
            print(f"LogisticRegression: {result['model_predictions']['logistic_regression']['prediction']}")
        
        # Save the label mapping for the web interface
        logger.info("Saving label mapping...")
        label_mapping_df = spark.createDataFrame([(k, v) for k, v in label_mapping.items()], ["label_id", "genre"])
        label_mapping_df.write.mode("overwrite").csv(f"{MODEL_BASE_PATH}_label_mapping", header=True)
        
        logger.info("\nModel training and evaluation complete!")
        print(f"Models saved to {MODEL_BASE_PATH}_rf, {MODEL_BASE_PATH}_gbt, and {MODEL_BASE_PATH}_lr")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")


'''

def main():
    """Main training function."""
    logger = setup_logging()
    logger.info("Starting music genre classifier training")
    
    try:
        # Check if dataset exists
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset not found at {DATASET_PATH}")
            logger.info("Please download the Mendeley music dataset and place it in the data directory")
            return
        
        # Create Spark session with increased memory like in original script
        spark = create_spark_session()
        
        # Initialize components
        data_processor = DataProcessor(spark)
        feature_engineer = FeatureEngineering(spark)
        model_trainer = ModelTrainer(spark)
        model_evaluator = ModelEvaluator(spark)
        
        # Load data
        logger.info("Loading data...")
        raw_df = data_processor.load_data(DATASET_PATH)
        
        # Display schema and sample data like in original script
        print("Dataset Schema:")
        raw_df.printSchema()
        print("\nSample Data:")
        raw_df.show(5, truncate=False)
        
        # Process and clean data
        logger.info("Cleaning data...")
        cleaned_df = data_processor.clean_data(raw_df)
        
        # Display genre distribution like in original script
        print("\nGenre Distribution:")
        genre_counts = cleaned_df.groupBy('genre').count().orderBy('count', ascending=False)
        genre_counts.show()
        
        # Balance dataset
        logger.info("Balancing dataset...")
        # balanced_df = data_processor.balance_dataset(cleaned_df, genre_counts)
        balanced_df = data_processor.balance_dataset(cleaned_df)
        
        # Show new distribution like in original script
        print("\nBalanced Genre Distribution:")
        balanced_df.groupBy('genre').count().orderBy('count', ascending=False).show()
        
        # Extract text features
        logger.info("Extracting text features...")
        featured_df = feature_engineer.add_text_features(cleaned_df)

        # Create train and test splits excluding the first column
        # Split data
        train_df, test_df = data_processor.split_data(featured_df)
        
        print("Data split and saved to train_data.csv and test_data.csv")

        print(f"\nTraining dataset size: {train_df.count()}")
        print(f"Testing dataset size: {test_df.count()}")
        
        # Train models
        logger.info("Training models...")
        
        # First create the pipelines
        print("\nTraining RandomForest model...")
        rf_pipeline = model_trainer.create_random_forest_pipeline()
        rf_model = rf_pipeline.fit(train_df)
        
        print("\nTraining GBT model...")
        gbt_pipeline = model_trainer.create_gbt_pipeline()
        gbt_model = gbt_pipeline.fit(train_df)
        
        print("\nTraining LogisticRegression model...")
        lr_pipeline = model_trainer.create_logistic_regression_pipeline()
        lr_model = lr_pipeline.fit(train_df)
        
        # Save models
        logger.info("Saving models...")
        rf_path = f"{MODEL_BASE_PATH}_rf"
        gbt_path = f"{MODEL_BASE_PATH}_gbt"
        lr_path = f"{MODEL_BASE_PATH}_lr"
        label_mapping_path = f"{MODEL_BASE_PATH}_label_mapping"
        
        # Call save_models with explicit paths
        model_trainer.save_models(
            rf_model, 
            gbt_model, 
            lr_model,
            rf_path=rf_path,
            gbt_path=gbt_path,
            lr_path=lr_path,
            label_mapping_path=label_mapping_path
        )
        
        # Evaluate individual models
        logger.info("Evaluating models...")
        rf_predictions = model_evaluator.evaluate_model(rf_model, test_df, "RandomForest")
        gbt_predictions = model_evaluator.evaluate_model(gbt_model, test_df, "GradientBoostedTrees")
        lr_predictions = model_evaluator.evaluate_model(lr_model, test_df, "LogisticRegression")
        
        # Get label mapping from the indexer and print
        label_mapping = {float(idx): genre for idx, genre in enumerate(rf_model.stages[0].labels)}
        print("\nLabel mapping:", label_mapping)
        
        # Create ensemble predictions using majority voting
        logger.info("Creating ensemble predictions using majority voting...")
        ensemble_preds = model_evaluator.create_majority_vote_ensemble(
            rf_predictions, gbt_predictions, lr_predictions
        )
        
        # Evaluate ensemble model
        model_evaluator.evaluate_ensemble(ensemble_preds)
        
        # Compare model performance
        logger.info("Comparing model performance...")
        model_evaluator.compare_models(ensemble_preds)
        
        # Display confusion matrix for the ensemble model
        print("\nEnsemble Model Confusion Matrix:")
        confusion_matrix = ensemble_preds.groupBy("label", "prediction").count().orderBy("label", "prediction")
        confusion_matrix.show()
        
        # Test the classifier with examples (like in original script)
        print("\n--- Testing the ensemble classifier with example lyrics ---")
        examples = [
            "I feel the rhythm in my soul as I dance through the night with stars above",
            "Broken hearts and whiskey, driving down these country roads thinking of you",
            "The beat drops hard, I'm spitting rhymes in the cipher, mic check one two",
            "Swinging notes and saxophone, the band plays jazz until dawn breaks through",
            "I got the blues in my heart, guitar crying with me all night long",
            "Rock and roll all night, guitar solos blazing as the crowd goes wild",
            "Jamming to the reggae beat, island vibes and sunshine feeling alright"
        ]
        
        classifier = model_evaluator.create_classifier(rf_model, gbt_model, lr_model)
        for example in examples:
            result = classifier.classify_lyrics(example)
            
            print(f"\nExample lyrics: '{example}'")
            print(f"Ensemble prediction: {result['ensemble_prediction']}")
            
            print("\nTop 3 genres by ensemble probability:")
            i = 0
            for genre, prob in sorted(result['ensemble_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"{genre}: {prob:.4f}")
                i += 1
                if i >= 3:
                    break
            
            print("\nIndividual model predictions:")
            print(f"RandomForest: {result['model_predictions']['random_forest']['prediction']}")
            print(f"GradientBoosting: {result['model_predictions']['gradient_boosting']['prediction']}")
            print(f"LogisticRegression: {result['model_predictions']['logistic_regression']['prediction']}")
        
        # The label mapping is already saved by the save_models() method
        logger.info("Label mapping saved to the specified path")
        
        logger.info("\nModel training and evaluation complete!")
        print(f"Models saved to {MODEL_BASE_PATH}_rf, {MODEL_BASE_PATH}_gbt, and {MODEL_BASE_PATH}_lr")
        
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}", exc_info=True)
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")
            
if __name__ == "__main__":
    main()

