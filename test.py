"""
Script for testing the music genre classifier with new lyrics.
"""

import logging
import os
import sys
import argparse

from src.models.genre_classifier import LyricsClassifier
from src.utils.spark_utils import create_spark_session
from config.app_config import MODEL_BASE_PATH


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('testing.log')
        ]
    )
    return logging.getLogger(__name__)


def load_models(spark):
    """Load the trained models."""
    from pyspark.ml import PipelineModel
    
    logger = logging.getLogger(__name__)
    
    # Define model paths
    rf_path = f"{MODEL_BASE_PATH}_rf"
    gbt_path = f"{MODEL_BASE_PATH}_gbt"
    lr_path = f"{MODEL_BASE_PATH}_lr"
    
    # Check if models exist
    for path in [rf_path, gbt_path, lr_path]:
        if not os.path.exists(path):
            logger.error(f"Model not found at {path}")
            logger.info("Please train the models first using train.py")
            return None, None, None
    
    # Load models
    logger.info("Loading models...")
    rf_model = PipelineModel.load(rf_path)
    gbt_model = PipelineModel.load(gbt_path)
    lr_model = PipelineModel.load(lr_path)
    
    return rf_model, gbt_model, lr_model


def test_with_examples(classifier):
    """Test the classifier with predefined examples."""
    examples = [
        "I feel the rhythm in my soul as I dance through the night with stars above",
        "Broken hearts and whiskey, driving down these country roads thinking of you",
        "The beat drops hard, I'm spitting rhymes in the cipher, mic check one two",
        "Swinging notes and saxophone, the band plays jazz until dawn breaks through",
        "I got the blues in my heart, guitar crying with me all night long",
        "Rock and roll all night, guitar solos blazing as the crowd goes wild",
        "Jamming to the reggae beat, island vibes and sunshine feeling alright"
    ]
    
    print("\n--- Testing with example lyrics ---")
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


def test_with_input(classifier):
    """Test the classifier with user input."""
    print("\n--- Enter lyrics to classify (type 'exit' to quit) ---")
    
    while True:
        lyrics = input("\nEnter lyrics: ")
        if lyrics.lower() == 'exit':
            break
            
        if not lyrics.strip():
            print("Please enter some lyrics to classify.")
            continue
            
        result = classifier.classify_lyrics(lyrics)
        
        print(f"\nEnsemble prediction: {result['ensemble_prediction']}")
        
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


def main():
    """Main testing function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test music genre classifier")
    parser.add_argument("--examples", action="store_true", help="Test with predefined examples")
    parser.add_argument("--input", action="store_true", help="Test with user input")
    args = parser.parse_args()
    
    # Default to both if no arguments specified
    if not args.examples and not args.input:
        args.examples = True
        args.input = True
    
    logger = setup_logging()
    logger.info("Starting music genre classifier testing")
    
    try:
        # Create Spark session
        spark = create_spark_session(
            app_name="Music Genre Classification Testing",
            driver_memory="4g",
            executor_memory="4g"
        )
        
        # Load models
        rf_model, gbt_model, lr_model = load_models(spark)
        if rf_model is None or gbt_model is None or lr_model is None:
            return
        
        # Create classifier
        classifier = LyricsClassifier(spark, rf_model, gbt_model, lr_model)
        
        # Test with examples if requested
        if args.examples:
            test_with_examples(classifier)
        
        # Test with user input if requested
        if args.input:
            test_with_input(classifier)
        
        logger.info("Testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in testing process: {str(e)}", exc_info=True)
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()