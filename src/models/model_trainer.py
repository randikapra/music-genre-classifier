"""
Model training functionality for the music genre classifier.
"""

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier, OneVsRest
import logging

from src.features.feature_engineering import FeatureEngineering
from config.app_config import RF_MODEL_PATH, GBT_MODEL_PATH, LR_MODEL_PATH, LABEL_MAPPING_PATH


class ModelTrainer:
    """Class for training models."""
    
    def __init__(self, spark):
        """
        Initialize with a SparkSession.
        
        Args:
            spark (SparkSession): The active Spark session
        """
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        self.feature_engineering = FeatureEngineering(spark)
    
    def create_random_forest_pipeline(self):
        """
        Create Random Forest classifier pipeline.
        
        Returns:
            Pipeline: Configured Random Forest pipeline
        """
        self.logger.info("Creating Random Forest pipeline")
        
        # Get feature engineering stages
        stages = self.feature_engineering.get_feature_pipeline_stages()
        
        # Add Random Forest classifier
        rf = RandomForestClassifier(
            labelCol="label", 
            featuresCol="features", 
            numTrees=300,
            maxDepth=25,
            minInstancesPerNode=2,
            bootstrap=True,
            featureSubsetStrategy="sqrt"
        )
        
        stages.append(rf)
        
        return Pipeline(stages=stages)
    
    def create_gbt_pipeline(self):
        """
        Create Gradient Boosted Trees classifier pipeline.
        
        Returns:
            Pipeline: Configured GBT pipeline
        """
        self.logger.info("Creating GBT pipeline")
        
        # Get feature engineering stages
        stages = self.feature_engineering.get_feature_pipeline_stages()
        
        # Create binary GBT classifier
        gbt_binary = GBTClassifier(
            labelCol="label", 
            featuresCol="features",
            maxIter=100,
            maxDepth=8,
            stepSize=0.1
        )
        
        # Wrap with OneVsRest for multiclass
        gbt = OneVsRest(classifier=gbt_binary)
        
        stages.append(gbt)
        
        return Pipeline(stages=stages)
    
    def create_logistic_regression_pipeline(self):
        """
        Create Logistic Regression classifier pipeline.
        
        Returns:
            Pipeline: Configured Logistic Regression pipeline
        """
        self.logger.info("Creating Logistic Regression pipeline")
        
        # Get feature engineering stages
        stages = self.feature_engineering.get_feature_pipeline_stages()
        
        # Add Logistic Regression classifier
        lr = LogisticRegression(
            labelCol="label", 
            featuresCol="features",
            maxIter=100,
            regParam=0.1,
            elasticNetParam=0.5
        )
        
        stages.append(lr)
        
        return Pipeline(stages=stages)
    
    def train_models(self, train_df):
        """
        Train all models.
        
        Args:
            train_df (DataFrame): Training dataset
            
        Returns:
            tuple: (rf_model, gbt_model, lr_model)
        """
        self.logger.info("Training all models")
        
        # Create pipelines
        rf_pipeline = self.create_random_forest_pipeline()
        gbt_pipeline = self.create_gbt_pipeline()
        lr_pipeline = self.create_logistic_regression_pipeline()
        
        # Train models
        self.logger.info("Training Random Forest model...")
        rf_model = rf_pipeline.fit(train_df)
        
        self.logger.info("Training GBT model...")
        gbt_model = gbt_pipeline.fit(train_df)
        
        self.logger.info("Training Logistic Regression model...")
        lr_model = lr_pipeline.fit(train_df)
        
        return rf_model, gbt_model, lr_model

    def save_models(self, rf_model, gbt_model, lr_model, rf_path=None, gbt_path=None, lr_path=None, label_mapping_path=None):
        """
        Save trained models to specified paths or default paths.
        
        Args:
            rf_model: Trained Random Forest model
            gbt_model: Trained GBT model
            lr_model: Trained Logistic Regression model
            rf_path: Path to save Random Forest model (optional)
            gbt_path: Path to save GBT model (optional)
            lr_path: Path to save Logistic Regression model (optional)
            label_mapping_path: Path to save label mapping (optional)
        """
        self.logger.info("Saving models...")
        
        # Use provided paths or fall back to constants
        rf_save_path = rf_path if rf_path else RF_MODEL_PATH
        gbt_save_path = gbt_path if gbt_path else GBT_MODEL_PATH
        lr_save_path = lr_path if lr_path else LR_MODEL_PATH
        label_save_path = label_mapping_path if label_mapping_path else LABEL_MAPPING_PATH
        
        try:
            rf_model.write().overwrite().save(rf_save_path)
            gbt_model.write().overwrite().save(gbt_save_path)
            lr_model.write().overwrite().save(lr_save_path)
            
            # Save label mapping for later use
            label_mapping = {float(idx): genre for idx, genre in enumerate(rf_model.stages[0].labels)}
            label_mapping_df = self.spark.createDataFrame(
                [(k, v) for k, v in label_mapping.items()], 
                ["label_id", "genre"]
            )
            label_mapping_df.write.mode("overwrite").csv(label_save_path, header=True)
            
            self.logger.info(f"Models saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise