import os
import numpy as np
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import mlflow
import mlflow.tensorflow

from src.config import config
from src.data.data_loader import download_dataset, create_data_generators, print_class_distribution
from src.models.model import create_model, define_metrics, get_callbacks, get_predictions_and_labels
from src.utils.visualization import plot_training_history, plot_confusion_matrix, plot_roc_and_prc

def main():
    # Set up MLflow
    mlflow.set_experiment("skin_cancer_prediction")
    
    # Override EPOCHS from environment variable if set (for CI)
    epochs = int(os.getenv("EPOCHS", config.EPOCHS))
    
    # Data preparation
    data_path = download_dataset()
    train_gen, val_gen, test_gen = create_data_generators(data_path)
    print_class_distribution([train_gen, val_gen, test_gen], ['train', 'validation', 'test'])

    # Calculate initial bias
    count_malignant = (train_gen.classes == 1).sum()
    count_benign = (train_gen.classes == 0).sum()
    output_bias = -np.log(count_malignant / count_benign)

    # Model setup
    with mlflow.start_run():
        model = create_model(output_bias)
        metrics = define_metrics()
        
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            metrics=metrics
        )

        # Training with MLflow logging
        mlflow.tensorflow.autolog()
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,  # Use variable here
            callbacks=get_callbacks()
        )

        # Evaluate and log test results
        test_results = model.evaluate(test_gen, return_dict=True)
        for metric_name, value in test_results.items():
            mlflow.log_metric(metric_name, value)

        # Predictions and visualization
        test_preds, test_labels = get_predictions_and_labels(model, test_gen)
        train_preds, train_labels = get_predictions_and_labels(model, train_gen)
        val_preds, val_labels = get_predictions_and_labels(model, val_gen)

        # Save plots
        plot_training_history(history.history)
        plot_confusion_matrix(test_labels, test_preds)
        plot_roc_and_prc(train_labels, train_preds, val_labels, val_preds, test_labels, test_preds)

        # Log plots as artifacts in MLflow
        mlflow.log_artifacts(config.PLOTS_DIR, artifact_path="plots")

if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(config.NUM_THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(config.NUM_THREADS)
    main()