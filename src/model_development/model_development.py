import os
import numpy as np
import pandas as pd
import time
from pathlib import Path

%pip install silence_tensorflow
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

%pip install tensorflow
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(16) 
tf.config.threading.set_inter_op_parallelism_threads(16)
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, SeparableConv2D
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras import layers

%pip install scikit-learn 
from sklearn import metrics

import matplotlib.pyplot as plt
%pip install seaborn 
import seaborn as sns
sns.set()

### Download dataset from kagglehub
%pip install kagglehub
import kagglehub

# Download latest version
path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")

print("Path to dataset files:", path)

train_dir = path + '/train/'
test_dir = path + '/test/'
print("Path to dataset files:", path)

#Genenrate image data sets for training, validation and test
image_size = (224, 224)
batch_size = 32

augmented_train_gen = ImageDataGenerator(
    rotation_range=30,
    brightness_range=[0.9,1.1],
    zoom_range=0.1,
    fill_mode='constant',
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    validation_split=0.2
)

validation_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = ImageDataGenerator(
    rescale=1./255
)

### training dataset

augmented_train_generator = augmented_train_gen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

augmented_validation_generator = validation_gen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

augmented_test_generator = test_gen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

for gen, name in zip([augmented_train_generator, augmented_validation_generator, augmented_test_generator],
                     ['train', 'validation', 'test']):
    count_malignant = (gen.classes==1).sum()
    count_benign = (gen.classes==0).sum()
    tot = len(gen.classes)
    print(f"Class distribution in the {name} set")
    print(f"Benign: {count_benign/tot:.0%}")
    print(f"Malignant: {count_malignant/tot:.0%}\n")

#### Define metrics

model_metrics = [
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.FalseNegatives(name='fn'),
]

output_bias = -np.log(count_malignant/count_benign)
print('Initial output bias:', output_bias)
output_bias = tf.keras.initializers.Constant(output_bias)

input_bias = -np.log(1/2)
print('Initiali output bias:', input_bias)
input_bias = tf.keras.initializers.Constant(input_bias)


#Donwload pre-trained weights
pre_trained_Xception = keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(None,None,3)
)

#pre_trained_Xception.summary()


### Change the model by setting trainable layers in Xception

pre_trained_Xception.trainable = True

for layer in pre_trained_Xception.layers:
  if 'block14_' in layer.name:
    layer.trainable = True
  else:
    layer.trainable = False


input_tensor = keras.Input(shape=(image_size+(3,)), name='input')
x = pre_trained_Xception(input_tensor)
x = keras.layers.Flatten(name='flatten')(x)
x = keras.layers.Dense(16, activation='relu', kernel_initializer='he_normal')(x)
output = keras.layers.Dense(1, activation='sigmoid', name='output', bias_initializer=output_bias)(x)

fine_turned_xception_model = keras.Model(inputs=input_tensor, outputs=output, name='fine_turned_xception_model')
#fine_turned_xception_model.summary()

### callback function
model_dir = 'Users/lenhanpham/model_dir'
os.makedirs(model_dir, exist_ok=True)
csv_dir = 'Users/lenhanpham/csv_dir'
os.makedirs(csv_dir, exist_ok=True)

def callbacks(model_name, append_csv=False):
  callb = [
      tf.keras.callbacks.ModelCheckpoint(
          filepath = model_dir + '/' + model_name + '.keras',
          monitor='val_auc',
          save_best_only=True,
          mode='max'
      ),
      tf.keras.callbacks.EarlyStopping(
          monitor='val_auc',
          patience=10,
          mode='max',
          restore_best_weights=True
      ),
      tf.keras.callbacks.CSVLogger(
          filename=csv_dir + '/' + model_name + '.csv',
          separator=',',
          append=append_csv
      )
  ]
  return callb


### compile the model and train it

fine_turned_xception_model.compile(loss='binary_crossentropy',
                                   optimizer=keras.optimizers.Adam(learning_rate=2e-5),
                                   metrics=model_metrics,
                                   )

fine_turned_xception_model_history = fine_turned_xception_model.fit(augmented_train_generator,
                                                                    validation_data=augmented_validation_generator,
                                                                    callbacks=callbacks('fine_turned_xception_model'),
                                                                    epochs=50)

def plot_training_history(metrics, hist_dict):
  epochs = range(1, len(hist_dict[metrics])+1)
  sns.scatterplot(x=epochs, y=hist_dict[metrics])
  sns.lineplot(x=epochs,y= hist_dict['val_'+metrics])
  plt.title(metrics)
  plt.xlabel('Epoch')
  plt.ylabel(metrics)
  plt.legend(['Train', 'Validation'])

def plot_4metrics(hist_dict):
  plt.figure(figsize=(12,8))
  metrics = ['accuracy', 'loss', 'auc', 'recall']
  for i, metric in enumerate(metrics):
    plt.subplot(2,2,i+1)
    plot_training_history(metric, hist_dict)
  plt.tight_layout()
  plt.show()

plot_4metrics(fine_turned_xception_model_history.history)



### Load the best model
trained_ftxception_model = tf.keras.models.load_model(model_dir + '/' + 'fine_turned_xception_model' + '.keras')

ft_test_results = trained_ftxception_model.evaluate(augmented_test_generator,return_dict=True)



### print out metrics
for metric_name, metric_value in ft_test_results.items():
    print(f'{metric_name:}: {metric_value:.2f}')


def get_predictions_and_labels(model, generator):
  prediction_testset = []
  label_testset = []
  num_batch = len(generator)
  i = 0
  for data, label in generator:
    prediction_testset.extend(model.predict(data).flatten())
    label_testset.extend(label)
    i +=1
    if i >= num_batch:
      prediction_testset = np.array(prediction_testset)
      label_testset = np.array(label_testset)
      break
  return prediction_testset, label_testset

test_preds_ftxception, test_labels_ftxception = get_predictions_and_labels(trained_ftxception_model, augmented_test_generator)



confustion_matrix = tf.math.confusion_matrix(labels=test_labels_ftxception, predictions=test_preds_ftxception>0.5)

## plot confusion matrix
sns.heatmap(confustion_matrix, annot=True, fmt='d')
plt.title('Confusion matrix @0.5')
plt.ylabel('True class')


### Plot ROC and AUC
train_preds_ftxception, train_labels_ftxception = get_predictions_and_labels(trained_ftxception_model, augmented_train_generator)
validation_preds_ftxception, validation_labels_ftxception = get_predictions_and_labels(trained_ftxception_model, augmented_validation_generator)


def plot_roc(name, labels, predictions, **kwargs):
    """ Function for plotting the ROC curve. """
    fp, tp, _ = metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 60])
    plt.ylim([20, 100.5])
    plt.grid(True)
    plt.title('ROC', fontsize=16)
    plt.legend(loc='lower right');


def plot_prc(name, labels, predictions, **kwargs):
    """ Function for plotting the area under the interpolated precision-recall curve (AUPRC). """
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.title('AUPRC', fontsize=16);


# Plot the ROC and AUPRC
plt.figure(figsize=(16,6))
plt.subplot(1, 2, 1)
plot_roc('Train xception', train_labels_ftxception, train_preds_ftxception, linestyle='--')
plot_roc('Validation xception', validation_labels_ftxception, validation_preds_ftxception, linestyle=':')
plot_roc('Test xception', test_labels_ftxception, test_preds_ftxception)

plt.subplot(1, 2, 2)
plot_prc('Train xception', train_labels_ftxception, train_preds_ftxception, linestyle='--')
plot_prc('Validation xception', validation_labels_ftxception, validation_preds_ftxception, linestyle=':')
plot_prc('Test xception', test_labels_ftxception, test_preds_ftxception)