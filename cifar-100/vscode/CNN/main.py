# Import TensorFlow, Keras components, and other utilities.
# - tensorflow as tf: The core TensorFlow library.
# - tensorflow.keras : TensorFlow's high-level API for building and training models.
# - layers: Module containing standard neural network layers (Conv2D, Dense, etc.).
# - models: Module for creating models (Sequential, Functional API).
# - datasets: Module containing built-in datasets like CIFAR-100.
# - optimizers: Module containing optimization algorithms (Adam, SGD, etc.).

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets, optimizers, losses
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

print("Libraries imported successfully.")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# Check for GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"GPU available: {gpu_devices}")
    # Optional: Configure GPU memory growth to avoid allocating all memory at once
    try:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth configured.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("GPU not available, using CPU.")

# Configuration and Hyperparameters
# - BATCH_SIZE: Number of images processed in one training step.
# - LEARNING_RATE: Controls the step size during optimization.
# - NUM_EPOCHS: How many times the entire training dataset is passed through the model.
# - NUM_CLASSES: CIFAR-100 has 100 distinct image categories.
# - INPUT_SHAPE: The dimensions of each input image (Height, Width, Channels).
# - L2_LAMBDA

BATCH_SIZE = 64          # Number of images per batch
LEARNING_RATE = 0.001    # Learning rate for the optimizer
NUM_EPOCHS = 500         # Number of times to iterate over the entire dataset
NUM_CLASSES = 100        # CIFAR-100 has 100 classes
INPUT_SHAPE = (32, 32, 3) # CIFAR images are 32x32 pixels with 3 color channels (RGB)
L2_LAMBDA = 0.0002      # Define L2 regularization strength

print(f"Configuration:")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Number of Epochs: {NUM_EPOCHS}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Input Shape: {INPUT_SHAPE}")

# Load the dataset directly using `tf.keras.datasets.cifar100`.
# This function returns NumPy arrays for training and testing images and labels.
# - Images (`x_train`, `x_test`) are NumPy arrays of shape (num_samples, 32, 32, 3) with pixel values in [0, 255].
# - Labels (`y_train`, `y_test`) are NumPy arrays of shape (num_samples, 1) containing integer labels from 0 to 99.

print("Loading CIFAR-100 dataset...")
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

print("Dataset loaded successfully.")
print(f"  x_train shape: {x_train.shape}") # (50000, 32, 32, 3)
# x_train holds 50,000 images. Each of those 50,000 images is a 32x32 grid of pixels.  
# And at each of those 32x32 pixel locations, there are 3 values representing the 
# Red, Green, and Blue color components of that pixel.

print(f"  y_train shape: {y_train.shape}") # (50000, 1)
print(f"  x_test shape: {x_test.shape}")   # (10000, 32, 32, 3)
print(f"  y_test shape: {y_test.shape}")   # (10000, 1)
print(f"  Number of training samples: {x_train.shape[0]}")
print(f"  Number of test samples: {x_test.shape[0]}")
print(f"  Image data type: {x_train.dtype}") # uint8
print(f"  Label data type: {y_train.dtype}") # int64
print(f"  Min/Max pixel values: {x_train.min()}/{x_train.max()}") # 0/255

# Prepare the data for training:
# - Convert Image Type: Change image data type from `uint8` to `float32` for calculations.
# - Normalize Pixels: Scale pixel values from the range [0, 255] to [0, 1]. This helps stabilize training. Alternatively, you could scale to [-1, 1] by dividing by 127.5 and subtracting 1.
# - Labels: The labels are already integers (0-99), which is the format expected by `SparseCategoricalCrossentropy` loss. No changes needed for `y_train`, `y_test`.

# Convert image data types to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize pixel values to the range [0, 1]
x_train /= 255.0
x_test /= 255.0

print(f"  x_train data type after conversion: {x_train.dtype}") # float32
print(f"  Min/Max pixel values after normalization: {x_train.min():.1f}/{x_train.max():.1f}") # 0.0/1.0

# Labels y_train and y_test remain as integer arrays of shape (N, 1)
print(f"  y_train shape remains: {y_train.shape}")

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=INPUT_SHAPE),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)

# Apply augmentation ONLY to the training data
# Apply augmentation within the tf.data pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train))\
                             .batch(BATCH_SIZE)\
                             .map(lambda x, y: (data_augmentation(x, training=True), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)\
                             .prefetch(tf.data.AUTOTUNE) # Add prefetching

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)\
                           .prefetch(tf.data.AUTOTUNE) # Add prefetching

# Define callbacks for training
# - EarlyStopping: Stop training when the validation loss stops improving to prevent overfitting
# - ReduceLROnPlateau: Reduce learning rate when the validation loss plateaus to help convergence
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Defining the Keras Sequential model (with Batch Norm, L2, Softmax output)...
print("Defining the Keras Sequential model (with Batch Norm, L2, Softmax output)...")

model = models.Sequential([
    # Block 1: Two Conv2D layers starting with 32 filters
    layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA), input_shape=INPUT_SHAPE),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Block 2: Three Conv2D layers with 64 filters
    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Block 3: Three Conv2D layers with 128 filters
    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Block 4: Two Conv2D layers with 256 filters
    layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),

    # Flatten and Dense Layers (expanded)
    layers.Flatten(),
    layers.Dense(2048, kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1024, kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

print("Model defined successfully with Batch Norm, L2, and Softmax output.")
model.summary()

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=NUM_EPOCHS,
    validation_data=test_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluating the trained model's performance on the test dataset using `model.evaluate()`.
# - Pass the test data (`test_dataset`).
# - It returns the final loss and metric values (e.g., accuracy) calculated on the test set.
# Evaluate the model

print("\nEvaluating the model on the test dataset...")

loss, accuracy = model.evaluate(
    test_dataset, 
    verbose=1
)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

print("\nPlotting training history...")

plt.plot(history.history['accuracy'], label='Training Accuracy') 
plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy over Epochs') 
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss over Epochs') 
plt.legend(loc='upper right')
plt.grid(True) 
plt.show()