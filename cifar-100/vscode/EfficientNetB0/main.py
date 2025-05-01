import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Hyperparameters
BATCH_SIZE    = 64
INPUT_SIZE    = 224         # 224×224 for EfficientNetB0 :contentReference[oaicite:4]{index=4}
NUM_CLASSES   = 100
LR_HEAD       = 1e-3
LR_FINE       = 1e-5
EPOCHS_HEAD   = 25
EPOCHS_FINE   = 20
L2_LAMBDA     = 2e-4

# Load CIFAR-100
(x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

# Preprocessing: resize only (no manual normalization)
def preprocess(image, label):
    image = tf.image.resize(image, [INPUT_SIZE, INPUT_SIZE])
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                   .shuffle(50000) \
                   .batch(BATCH_SIZE)

# Data augmentation after resizing
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
                        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE) \
                        .batch(BATCH_SIZE) \
                        .prefetch(tf.data.AUTOTUNE)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr      = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# Build model with EfficientNetB0 backbone (includes Rescaling layer) :contentReference[oaicite:5]{index=5}
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(INPUT_SIZE, INPUT_SIZE, 3)
)
base_model.trainable = False  # Phase 1

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu', kernel_regularizer=l2(L2_LAMBDA)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_HEAD),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Phase 1: Train only the head
history_head = model.fit(
    train_ds,
    epochs=EPOCHS_HEAD,
    validation_data=test_ds,
    callbacks=[early_stopping, reduce_lr]
)

# Phase 2: Unfreeze last 20 layers (except BatchNorm) for fine-tuning :contentReference[oaicite:6]{index=6}
for layer in base_model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_FINE),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=test_ds,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate final performance
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy*100:.2f}%")

# Plot training history
plt.figure()
plt.plot(history_head.history['accuracy']  + history_fine.history['accuracy'],  label='Train Acc')
plt.plot(history_head.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy'); plt.show()

plt.figure()
plt.plot(history_head.history['loss']      + history_fine.history['loss'],      label='Train Loss')
plt.plot(history_head.history['val_loss']  + history_fine.history['val_loss'],  label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss'); plt.show()
