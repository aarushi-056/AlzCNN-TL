import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Use non-interactive backend for matplotlib (for headless environments)
matplotlib.use('Agg')


# Load and preprocess images
def load_images_from_directory(directory, target_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))  # Assumes subfolders represent classes
    class_map = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, filename)
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img)
                # Apply skull stripping
                img_array = skull_strip(img_array)
                images.append(img_array)
                labels.append(class_map[class_name])

    return np.array(images), np.array(labels), class_names


# Skull stripping function
def skull_strip(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, 11)

    brain_only = cv2.bitwise_and(image, image, mask=mask)
    return brain_only


# Z-score normalization
def z_score_normalize(images):
    images = np.asarray(images, dtype=np.float32)
    mean = np.mean(images, axis=(1, 2, 3), keepdims=True)
    std = np.std(images, axis=(1, 2, 3), keepdims=True)
    return (images - mean) / (std + 1e-8)


# CNN model definition
def build_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrixMod1.png")
    plt.close()


# Plot combined training and validation metrics for all folds
def plot_combined_metrics(training_accuracy, validation_accuracy, training_loss, validation_loss):
    epochs = range(1, len(training_accuracy) + 1)

    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_accuracy, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Overall Training and Validation Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Overall Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('overall_training_validation_metricsMod1.png')  # Save the combined metrics plot
    plt.close()
    print("Overall training and validation metrics plot saved as 'overall_training_validation_metrics.png'.")


# Main cross-validation function
def cross_validate_train_test(X_train, y_train, X_test, y_test, n_splits=10, epochs=50, batch_size=32, num_classes=2):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true_test = []
    all_y_pred_test = []

    # To store metrics across folds
    training_accuracy_per_epoch = []
    validation_accuracy_per_epoch = []
    training_loss_per_epoch = []
    validation_loss_per_epoch = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Split data
        X_fold_train, X_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_val = y_train[train_idx], y_train[val_idx]

        # Normalize data
        X_fold_train = z_score_normalize(X_fold_train)
        X_val = z_score_normalize(X_val)
        X_test_normalized = z_score_normalize(X_test)

        # Convert labels to one-hot encoding
        y_fold_train_one_hot = tf.keras.utils.to_categorical(y_fold_train, num_classes)
        y_val_one_hot = tf.keras.utils.to_categorical(y_val, num_classes)

        # Build and train the model
        model = build_model(input_shape=X_fold_train.shape[1:], num_classes=num_classes)
        history = model.fit(
            X_fold_train, y_fold_train_one_hot,
            validation_data=(X_val, y_val_one_hot),
            epochs=epochs, batch_size=batch_size, verbose=1
        )

        # Aggregate metrics for plotting later
        if fold == 0:  # Initialize lists on the first fold
            training_accuracy_per_epoch = np.array(history.history['accuracy'])
            validation_accuracy_per_epoch = np.array(history.history['val_accuracy'])
            training_loss_per_epoch = np.array(history.history['loss'])
            validation_loss_per_epoch = np.array(history.history['val_loss'])
        else:  # Add to existing lists
            training_accuracy_per_epoch += np.array(history.history['accuracy'])
            validation_accuracy_per_epoch += np.array(history.history['val_accuracy'])
            training_loss_per_epoch += np.array(history.history['loss'])
            validation_loss_per_epoch += np.array(history.history['val_loss'])

        # Test the model on the separate test set
        y_pred_test = np.argmax(model.predict(X_test_normalized), axis=1)
        all_y_true_test.extend(y_test)
        all_y_pred_test.extend(y_pred_test)

    # Average metrics over folds
    training_accuracy_per_epoch /= n_splits
    validation_accuracy_per_epoch /= n_splits
    training_loss_per_epoch /= n_splits
    validation_loss_per_epoch /= n_splits

    # Final classification report on test set
    print("\nClassification Report (Test Set):")
    print(classification_report(all_y_true_test, all_y_pred_test, target_names=train_class_names))

    # Plot confusion matrix
    plot_confusion_matrix(all_y_true_test, all_y_pred_test, train_class_names)

    # Plot combined training and validation metrics
    plot_combined_metrics(
        training_accuracy_per_epoch, validation_accuracy_per_epoch,
        training_loss_per_epoch, validation_loss_per_epoch
    )


# Main script
if __name__ == "__main__":
    # Specify dataset path
    dataset_path = "/home/vidur/aarushi/dataset Model 1"
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    # Load train and test sets
    X_train, y_train, train_class_names = load_images_from_directory(train_path)
    X_test, y_test, test_class_names = load_images_from_directory(test_path)

    # Check consistency of classes
    assert train_class_names == test_class_names, "Class names mismatch between train and test sets."

    # Perform cross-validation and evaluation
    cross_validate_train_test(X_train, y_train, X_test, y_test, num_classes=len(train_class_names))

