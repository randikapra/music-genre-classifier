'''
"""
Web application for the music genre classifier.
"""

import os
import logging
import json
from flask import Flask, render_template, request, jsonify
import sys

from src.utils.spark_utils import create_spark_session
from src.models.genre_classifier import GenreClassifier
from config.app_config import WEB_HOST, WEB_PORT, DEBUG_MODE, CHART_COLORS


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# Check if models exist before starting the app
from config.app_config import RF_MODEL_PATH, GBT_MODEL_PATH, LR_MODEL_PATH
if not (os.path.exists(RF_MODEL_PATH) and 
        os.path.exists(GBT_MODEL_PATH) and 
        os.path.exists(LR_MODEL_PATH)):
    logger.error("Models not found. Please run train.py first.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize Spark and load the classifier
logger.info("Initializing Spark session and loading models...")
spark = create_spark_session()
classifier = GenreClassifier(spark)
logger.info("Models loaded successfully")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', chart_colors=json.dumps(CHART_COLORS))


@app.route('/api/classify', methods=['POST'])
def classify_lyrics():
    """API endpoint for classifying lyrics."""
    try:
        # Get data from request
        data = request.get_json()
        lyrics = data.get('lyrics', '')
        year = data.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': {
                'random_forest': result['model_predictions']['random_forest']['prediction'],
                'gradient_boosting': result['model_predictions']['gradient_boosting']['prediction'],
                'logistic_regression': result['model_predictions']['logistic_regression']['prediction']
            }
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    try:
        logger.info(f"Starting web server on {WEB_HOST}:{WEB_PORT}")
        app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG_MODE)
    except KeyboardInterrupt:
        logger.info("Shutting down web server")
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")
'''


### --version 2.0-- ###
'''
import os
import sys
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, udf

# Configuration settings
RF_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_rf"
GBT_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_gbt"
LR_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_lr"
LABEL_MAPPING_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_label_mapping"

# Web server settings
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
DEBUG_MODE = False

# Chart colors for visualization
CHART_COLORS = {
    'pop': '#FF6384',      # Pink
    'country': '#36A2EB',  # Blue
    'blues': '#FFCE56',    # Yellow
    'jazz': '#4BC0C0',     # Teal
    'reggae': '#9966FF',   # Purple
    'rock': '#FF9F40',     # Orange
    'hip hop': '#C9CBCF'   # Gray
}

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("Music Genre Classifier Web App") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

class GenreClassifier:
    """Class for music genre classification using ensemble model."""
    
    def __init__(self, spark_session):
        """Initialize the classifier by loading the trained models."""
        self.spark = spark_session
        
        # Load models
        self.rf_model = PipelineModel.load(RF_MODEL_PATH)
        self.gbt_model = PipelineModel.load(GBT_MODEL_PATH)
        self.lr_model = PipelineModel.load(LR_MODEL_PATH)
        
        # Get label mapping from first model
        self.label_mapping = {float(idx): genre for idx, genre in enumerate(self.rf_model.stages[0].labels)}
        
        # Create UDF for text length feature
        self.text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
    
    def classify_lyrics(self, lyrics_text, year=2025):
        """
        Classify new lyrics using the ensemble of trained models.
        Returns the predicted genre and probabilities from all models.
        """
        # Create a DataFrame with the new lyrics
        schema = StructType([
            StructField("artist_name", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("genre", StringType(), True),
            StructField("lyrics", StringType(), True),
            StructField("year", IntegerType(), True),
        ])
        
        new_data = self.spark.createDataFrame(
            [(None, None, str(year), "unknown", lyrics_text, year)],
            schema=schema
        )
        
        # Add text length feature
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
        num_classes = len(self.label_mapping)
        gbt_probs = np.zeros(num_classes)
        gbt_probs[int(gbt_label)] = 1.0  # Set 1.0 for the predicted class

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
        rf_genre_probs = {self.label_mapping[float(i)]: float(rf_probs[i]) for i in range(len(rf_probs)) if float(i) in self.label_mapping}
        gbt_genre_probs = {self.label_mapping[float(i)]: float(gbt_probs[i]) for i in range(len(gbt_probs)) if float(i) in self.label_mapping}
        lr_genre_probs = {self.label_mapping[float(i)]: float(lr_probs[i]) for i in range(len(lr_probs)) if float(i) in self.label_mapping}
        
        # Calculate weighted ensemble probabilities
        rf_weight = 0.40
        gbt_weight = 0.35
        lr_weight = 0.25
        
        ensemble_probs = {}
        for i in range(len(self.label_mapping)):
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

# Initialize logging
logger = setup_logging()

# Check if models exist before starting the app
if not (os.path.exists(RF_MODEL_PATH) and 
        os.path.exists(GBT_MODEL_PATH) and 
        os.path.exists(LR_MODEL_PATH)):
    logger.error("Models not found. Please run train.py first.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize Spark and load the classifier
logger.info("Initializing Spark session and loading models...")
spark = create_spark_session()
classifier = GenreClassifier(spark)
logger.info("Models loaded successfully")

@app.route('/')
def index():
    """Render the main page."""
    # return render_template('index.html', chart_colors=json.dumps(CHART_COLORS))
    return render_template('index.html')  # Remove the chart_colors parameter

@app.route('/classify', methods=['POST'])
def classify_lyrics():
    """API endpoint for classifying lyrics."""
    try:
        # Get data from request
        lyrics = request.form.get('lyrics', '')
        year = request.form.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': {
                'random_forest': result['model_predictions']['random_forest']['prediction'],
                'gradient_boosting': result['model_predictions']['gradient_boosting']['prediction'],
                'logistic_regression': result['model_predictions']['logistic_regression']['prediction']
            }
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify_lyrics():
    """API endpoint for classifying lyrics (JSON input)."""
    try:
        # Get data from request
        data = request.get_json()
        lyrics = data.get('lyrics', '')
        year = data.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': {
                'random_forest': result['model_predictions']['random_forest']['prediction'],
                'gradient_boosting': result['model_predictions']['gradient_boosting']['prediction'],
                'logistic_regression': result['model_predictions']['logistic_regression']['prediction']
            }
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

@app.route('/api/chart-colors')
def get_chart_colors():
    return jsonify(CHART_COLORS)

if __name__ == '__main__':
    try:
        logger.info(f"Starting web server on {WEB_HOST}:{WEB_PORT}")
        app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG_MODE)
    except KeyboardInterrupt:
        logger.info("Shutting down web server")
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")
'''


### --version 3.0-- ###
'''
import os
import sys
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, udf

# Configuration settings
RF_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_rf"
GBT_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_gbt"
LR_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_lr"
LABEL_MAPPING_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_label_mapping"

# Web server settings
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
DEBUG_MODE = False

# Chart colors for visualization
CHART_COLORS = {
    'pop': '#FF6384',      # Pink
    'country': '#36A2EB',  # Blue
    'blues': '#FFCE56',    # Yellow
    'jazz': '#4BC0C0',     # Teal
    'reggae': '#9966FF',   # Purple
    'rock': '#FF9F40',     # Orange
    'hip hop': '#C9CBCF'   # Gray
}

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("Music Genre Classifier Web App") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

class GenreClassifier:
    """Class for music genre classification using ensemble model."""
    
    def __init__(self, spark_session):
        """Initialize the classifier by loading the trained models."""
        self.spark = spark_session
        
        # Load models
        self.rf_model = PipelineModel.load(RF_MODEL_PATH)
        self.gbt_model = PipelineModel.load(GBT_MODEL_PATH)
        self.lr_model = PipelineModel.load(LR_MODEL_PATH)
        
        # Get label mapping from first model
        self.label_mapping = {float(idx): genre for idx, genre in enumerate(self.rf_model.stages[0].labels)}
        
        # Create UDF for text length feature
        self.text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
    
    def classify_lyrics(self, lyrics_text, year=2025):
        """
        Classify new lyrics using the ensemble of trained models.
        Returns the predicted genre and probabilities from all models.
        """
        # Create a DataFrame with the new lyrics
        schema = StructType([
            StructField("artist_name", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("genre", StringType(), True),
            StructField("lyrics", StringType(), True),
            StructField("year", IntegerType(), True),
        ])
        
        new_data = self.spark.createDataFrame(
            [(None, None, str(year), "unknown", lyrics_text, year)],
            schema=schema
        )
        
        # Add text length feature
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
        num_classes = len(self.label_mapping)
        gbt_probs = np.zeros(num_classes)
        gbt_probs[int(gbt_label)] = 1.0  # Set 1.0 for the predicted class

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
        rf_genre_probs = {self.label_mapping[float(i)]: float(rf_probs[i]) for i in range(len(rf_probs)) if float(i) in self.label_mapping}
        gbt_genre_probs = {self.label_mapping[float(i)]: float(gbt_probs[i]) for i in range(len(gbt_probs)) if float(i) in self.label_mapping}
        lr_genre_probs = {self.label_mapping[float(i)]: float(lr_probs[i]) for i in range(len(lr_probs)) if float(i) in self.label_mapping}
        
        # Calculate weighted ensemble probabilities
        rf_weight = 0.40
        gbt_weight = 0.35
        lr_weight = 0.25
        
        ensemble_probs = {}
        for i in range(len(self.label_mapping)):
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
                "random_forest": rf_genre,
                "gradient_boosting": gbt_genre,
                "logistic_regression": lr_genre
            }
        }
        
        return result

# Initialize logging
logger = setup_logging()

# Check if models exist before starting the app
if not (os.path.exists(RF_MODEL_PATH) and 
        os.path.exists(GBT_MODEL_PATH) and 
        os.path.exists(LR_MODEL_PATH)):
    logger.error("Models not found. Please run train.py first.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize Spark and load the classifier
logger.info("Initializing Spark session and loading models...")
spark = create_spark_session()
classifier = GenreClassifier(spark)
logger.info("Models loaded successfully")

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/static/js/script.js')
def serve_script():
    """Explicitly serve the script.js file."""
    return send_from_directory('static/js', 'script.js')

@app.route('/classify', methods=['POST'])
def classify_lyrics():
    """API endpoint for classifying lyrics."""
    try:
        # Get data from request
        lyrics = request.form.get('lyrics', '')
        year = request.form.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': result['model_predictions']
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify_lyrics():
    """API endpoint for classifying lyrics (JSON input)."""
    try:
        # Get data from request
        data = request.get_json()
        lyrics = data.get('lyrics', '')
        year = data.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': result['model_predictions']
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.path}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"500 error: {str(e)}")
    return render_template('500.html'), 500

@app.route('/api/chart-colors')
def get_chart_colors():
    return jsonify(CHART_COLORS)

if __name__ == '__main__':
    try:
        logger.info(f"Starting web server on {WEB_HOST}:{WEB_PORT}")
        app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG_MODE)
    except KeyboardInterrupt:
        logger.info("Shutting down web server")
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")
'''

### --version 4.0-- ###

import os
import sys
import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, udf

# # Configuration settings
# RF_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_rf"
# GBT_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_gbt"
# LR_MODEL_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_lr"
# LABEL_MAPPING_PATH = "/home/oshadi/SISR-Final_Year_Project/envs/ZZBigData/music-genre-classifier1/models/music_genre_classifier_label_mapping"


import subprocess
from pathlib import Path
# Get the base directory (to make paths relative)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration settings with relative paths
RF_MODEL_PATH = os.path.join(BASE_DIR, "models/music_genre_classifier_rf")
GBT_MODEL_PATH = os.path.join(BASE_DIR, "models/music_genre_classifier_gbt")
LR_MODEL_PATH = os.path.join(BASE_DIR, "models/music_genre_classifier_lr")
LABEL_MAPPING_PATH = os.path.join(BASE_DIR, "models/music_genre_classifier_label_mapping")


# Web server settings
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
DEBUG_MODE = False

# Chart colors for visualization
CHART_COLORS = {
    'pop': '#FF6384',      # Pink
    'country': '#36A2EB',  # Blue
    'blues': '#FFCE56',    # Yellow
    'jazz': '#4BC0C0',     # Teal
    'reggae': '#9966FF',   # Purple
    'rock': '#FF9F40',     # Orange
    'hip hop': '#C9CBCF',   # Gray
    'K-pop': '#00FF00'  # Bright Green (New Addition)
}

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


def check_spark_availability():
    """Check if spark-shell is available as mentioned in requirements."""
    try:
        # Just check if spark-shell command exists
        subprocess.run(["which", "spark-shell"], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       check=True,
                       timeout=5)
        logging.info("spark-shell is available in the environment.")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.warning("spark-shell command not found. Assuming PySpark is configured correctly.")
        return False

def create_spark_session():
    """Create and return a Spark session."""
    from pyspark.sql import SparkSession
    
    # Check if spark-shell is available (as per requirement)
    check_spark_availability()

    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("Music Genre Classifier Web App") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

class GenreClassifier:
    """Class for music genre classification using ensemble model."""
    
    def __init__(self, spark_session):
        """Initialize the classifier by loading the trained models."""
        self.spark = spark_session
        
        # Load models
        self.rf_model = PipelineModel.load(RF_MODEL_PATH)
        self.gbt_model = PipelineModel.load(GBT_MODEL_PATH)
        self.lr_model = PipelineModel.load(LR_MODEL_PATH)
        
        # Get label mapping from first model
        self.label_mapping = {float(idx): genre for idx, genre in enumerate(self.rf_model.stages[0].labels)}
        
        # Create UDF for text length feature
        self.text_length_udf = udf(lambda text: len(text.split()) if text else 0, IntegerType())
    
    def classify_lyrics(self, lyrics_text, year=2025):
        """
        Classify new lyrics using the ensemble of trained models.
        Returns the predicted genre and probabilities from all models.
        """
        # Create a DataFrame with the new lyrics
        schema = StructType([
            StructField("artist_name", StringType(), True),
            StructField("track_name", StringType(), True),
            StructField("release_date", StringType(), True),
            StructField("genre", StringType(), True),
            StructField("lyrics", StringType(), True),
            StructField("year", IntegerType(), True),
        ])
        
        new_data = self.spark.createDataFrame(
            [(None, None, str(year), "unknown", lyrics_text, year)],
            schema=schema
        )
        
        # Add text length feature
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
        num_classes = len(self.label_mapping)
        gbt_probs = np.zeros(num_classes)
        gbt_probs[int(gbt_label)] = 1.0  # Set 1.0 for the predicted class

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
        rf_genre_probs = {self.label_mapping[float(i)]: float(rf_probs[i]) for i in range(len(rf_probs)) if float(i) in self.label_mapping}
        gbt_genre_probs = {self.label_mapping[float(i)]: float(gbt_probs[i]) for i in range(len(gbt_probs)) if float(i) in self.label_mapping}
        lr_genre_probs = {self.label_mapping[float(i)]: float(lr_probs[i]) for i in range(len(lr_probs)) if float(i) in self.label_mapping}
        
        # Calculate weighted ensemble probabilities
        rf_weight = 0.40
        gbt_weight = 0.35
        lr_weight = 0.25
        
        ensemble_probs = {}
        for i in range(len(self.label_mapping)):
            genre = self.label_mapping[float(i)]
            try:
                weighted_prob = (rf_probs[i] * rf_weight) + (gbt_probs[i] * gbt_weight) + (lr_probs[i] * lr_weight)
                ensemble_probs[genre] = float(weighted_prob)
            except IndexError:
                # Handle case where probabilities arrays might have different lengths
                ensemble_probs[genre] = 0.0
                logging.warning(f"Index error for genre {genre}, setting probability to 0")
        
        # Normalize ensemble probabilities
        total = sum(ensemble_probs.values())
        if total > 0:  # Prevent division by zero
            for genre in ensemble_probs:
                ensemble_probs[genre] /= total
        
        # Sort probabilities for better visualization (highest first)
        ensemble_probs = {k: v for k, v in sorted(ensemble_probs.items(), key=lambda item: item[1], reverse=True)}
        
        # Create a comprehensive result with all model predictions
        result = {
            "ensemble_prediction": ensemble_genre,
            "ensemble_probabilities": ensemble_probs,
            "model_predictions": {
                "random_forest": rf_genre,
                "gradient_boosting": gbt_genre,
                "logistic_regression": lr_genre
            }
        }
        
        return result

# Initialize logging
logger = setup_logging()

# Check if models exist before starting the app
if not (os.path.exists(RF_MODEL_PATH) and 
        os.path.exists(GBT_MODEL_PATH) and 
        os.path.exists(LR_MODEL_PATH)):
    logger.error("Models not found. Please run train.py first.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Initialize Spark and load the classifier
logger.info("Initializing Spark session and loading models...")
spark = create_spark_session()
classifier = GenreClassifier(spark)
logger.info("Models loaded successfully")

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', chart_colors=json.dumps(CHART_COLORS))

@app.route('/static/<path:path>')
def serve_static(path):
    """Explicitly serve static files."""
    return send_from_directory('static', path)

@app.route('/classify', methods=['POST'])
def classify_lyrics():
    """API endpoint for classifying lyrics."""
    try:
        # Get data from request
        lyrics = request.form.get('lyrics', '')
        year = request.form.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': result['model_predictions']
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify_lyrics():
    """API endpoint for classifying lyrics (JSON input)."""
    try:
        # Get data from request
        data = request.get_json()
        lyrics = data.get('lyrics', '')
        year = data.get('year', 2025)
        
        if not lyrics or len(lyrics.strip()) == 0:
            return jsonify({'error': 'No lyrics provided'}), 400
        
        # Try to convert year to integer
        try:
            year = int(year)
        except ValueError:
            year = 2025  # Default to current year if invalid
        
        # Classify the lyrics
        logger.info(f"Classifying lyrics (length: {len(lyrics)})")
        result = classifier.classify_lyrics(lyrics, year)
        
        # Format the result for the frontend
        response = {
            'ensemble_prediction': result['ensemble_prediction'],
            'ensemble_probabilities': result['ensemble_probabilities'],
            'model_predictions': result['model_predictions']
        }
        
        logger.info(f"Classification result: {result['ensemble_prediction']}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error classifying lyrics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.path}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"500 error: {str(e)}")
    return render_template('500.html'), 500

@app.route('/api/chart-colors')
def get_chart_colors():
    return jsonify(CHART_COLORS)

if __name__ == '__main__':
    try:
        logger.info(f"Starting web server on {WEB_HOST}:{WEB_PORT}")
        app.run(host=WEB_HOST, port=WEB_PORT, debug=DEBUG_MODE)
    except KeyboardInterrupt:
        logger.info("Shutting down web server")
    finally:
        if 'spark' in locals():
            spark.stop()
            logger.info("Spark session stopped")