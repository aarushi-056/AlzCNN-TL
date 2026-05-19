import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2B0
from sklearn.metrics import confusion_matrix
from math import ceil, sqrt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize the images to the [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Load the pre-trained EfficientNetV2B0 model without the top layer
efficient_net_v2 = EfficientNetV2B0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3), pooling="avg"
)

# Build the model
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 3)),
    layers.Resizing(224, 224),  # Resize CIFAR-10 images to 224x224 for EfficientNetV2
    efficient_net_v2,
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compile the model
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4, decay_steps=10000, decay_rate=0.9
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True
)
history = model.fit(
    x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping], batch_size=32
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}, accuracy: {accuracy:.4f}')

# Make predictions
predictions = model.predict(x_test)
true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion Matrix
confusion = confusion_matrix(true_labels, predicted_labels)
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Display misclassified images
misclassified_indices = np.where(predicted_labels != true_labels)[0]
np.random.shuffle(misclassified_indices)
num_images = min(len(misclassified_indices), 20)
num_cols = int(ceil(sqrt(num_images)))
num_rows = int(ceil(num_images / num_cols))

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))

for i, idx in enumerate(misclassified_indices[:num_images]):
    true_class = true_labels[idx]
    predicted_class = predicted_labels[idx]
    axs[i // num_cols, i % num_cols].imshow(x_test[idx])
    axs[i // num_cols, i % num_cols].set_title(
        f'True: {class_names[true_class]}\nPred: {class_names[predicted_class]}'
    )
    axs[i // num_cols, i % num_cols].axis('off')

for i in range(num_images, num_rows * num_cols):
    axs[i // num_cols, i % num_cols].axis('off')

fig.suptitle("Misclassified Images")
plt.tight_layout()
plt.show()
