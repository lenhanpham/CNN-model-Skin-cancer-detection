import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.config import config
import os

def create_model(output_bias):
    pre_trained_Xception = keras.applications.Xception(
        include_top=False, weights='imagenet', input_shape=(None, None, 3)
    )
    
    pre_trained_Xception.trainable = True
    for layer in pre_trained_Xception.layers:
        layer.trainable = 'block14_' in layer.name

    input_tensor = keras.Input(shape=(config.IMAGE_SIZE + (3,)), name='input')
    x = pre_trained_Xception(input_tensor)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(config.DENSE_UNITS, activation='relu', kernel_initializer='he_normal')(x)
    output = layers.Dense(1, activation='sigmoid', name='output', 
                         bias_initializer=tf.keras.initializers.Constant(output_bias))(x)

    model = keras.Model(inputs=input_tensor, outputs=output, name=config.MODEL_NAME)
    return model

def define_metrics():
    return [
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.FalseNegatives(name='fn'),
    ]

def get_callbacks():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODEL_DIR, f'{config.MODEL_NAME}.keras'),
            monitor='val_auc', save_best_only=True, mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=config.PATIENCE, mode='max', restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(config.LOGS_DIR, f'{config.MODEL_NAME}.csv'),
            separator=',', append=False
        )
    ]

def get_predictions_and_labels(model, generator):
    predictions, labels = [], []
    num_batches = len(generator)
    for i, (data, label) in enumerate(generator):
        predictions.extend(model.predict(data).flatten())
        labels.extend(label)
        if i >= num_batches - 1:
            break
    return np.array(predictions), np.array(labels)