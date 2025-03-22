import os
import shutil
from pathlib import Path
import kagglehub
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.config import config

def download_dataset():
    # Download dataset to temporary cache location
    temp_path = kagglehub.dataset_download(config.DATASET_NAME)
    
    # Define target directory
    target_dir = config.DATA_DIR  # From config.py, points to "data/raw"
    os.makedirs(target_dir, exist_ok=True)
    
    # Move downloaded dataset to target_dir
    if os.path.exists(temp_path):
        # If temp_path contains the dataset directly, move its contents
        for item in os.listdir(temp_path):
            src = os.path.join(temp_path, item)
            dst = os.path.join(target_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        
        # Clean up temporary directory if empty
        try:
            shutil.rmtree(temp_path)
        except OSError:
            pass  # Ignore if temp_path can't be deleted
    
    return target_dir

def create_data_generators(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    augmented_train_gen = ImageDataGenerator(
        rotation_range=config.ROTATION_RANGE,
        brightness_range=config.BRIGHTNESS_RANGE,
        zoom_range=config.ZOOM_RANGE,
        fill_mode=config.FILL_MODE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        vertical_flip=config.VERTICAL_FLIP,
        rescale=config.RESCALE,
        validation_split=config.VALIDATION_SPLIT
    )

    validation_gen = ImageDataGenerator(
        rescale=config.RESCALE,
        validation_split=config.VALIDATION_SPLIT
    )
    test_gen = ImageDataGenerator(rescale=config.RESCALE)

    train_generator = augmented_train_gen.flow_from_directory(
        train_dir, 
        target_size=config.IMAGE_SIZE, 
        batch_size=config.BATCH_SIZE,
        class_mode='binary', 
        subset='training'
    )
    
    validation_generator = validation_gen.flow_from_directory(
        train_dir, 
        target_size=config.IMAGE_SIZE, 
        batch_size=config.BATCH_SIZE,
        class_mode='binary', 
        subset='validation'
    )
    
    test_generator = test_gen.flow_from_directory(
        test_dir, 
        target_size=config.IMAGE_SIZE, 
        batch_size=config.BATCH_SIZE,
        class_mode='binary'
    )

    return train_generator, validation_generator, test_generator

def print_class_distribution(generators, names):
    for gen, name in zip(generators, names):
        count_malignant = (gen.classes == 1).sum()
        count_benign = (gen.classes == 0).sum()
        total = len(gen.classes)
        print(f"Class distribution in the {name} set")
        print(f"Benign: {count_benign/total:.0%}")
        print(f"Malignant: {count_malignant/total:.0%}\n")