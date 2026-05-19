import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os


# ── Preprocessing ────────────────────────────────────────────────────────────

def skull_strip(image):
    """Remove non-brain tissue from an MRI image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    mask = cv2.medianBlur(mask, 11)
    return cv2.bitwise_and(image, image, mask=mask)


def z_score_normalize(images):
    """Apply Z-score normalisation across each image."""
    images = np.asarray(images, dtype=np.float32)
    mean = np.mean(images, axis=(1, 2, 3), keepdims=True)
    std  = np.std(images,  axis=(1, 2, 3), keepdims=True)
    return (images - mean) / (std + 1e-8)


def load_images_from_directory(directory, target_size=(128, 128)):
    """Load images, apply skull stripping, and return arrays + labels."""
    images, labels = [], []
    class_names = sorted(os.listdir(directory))
    class_map = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = load_img(os.path.join(class_dir, filename),
                               target_size=target_size)
                img_array = img_to_array(img)
                img_array = skull_strip(img_array)
                images.append(img_array)
                labels.append(class_map[class_name])

    return np.array(images), np.array(labels), class_names


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(input_shape):
    """
    CNN architecture as described in the paper (Model 2):
      Input → Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool →
      Flatten → Dense(128) → Dropout(0.25) → Sigmoid output
    Binary classification using sigmoid activation and binary cross-entropy loss.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='sigmoid')           # Binary output
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrixMod2.png')
    plt.close()
    print("Confusion matrix saved as 'confusion_matrixMod2.png'.")


def plot_combined_metrics(train_acc, val_acc, train_loss, val_loss):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc,  label='Training Accuracy')
    plt.plot(epochs, val_acc,    label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss,   label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_validation_metricsMod2.png')
    plt.close()
    print("Metrics plot saved as 'training_validation_metricsMod2.png'.")


# ── 10-Fold Cross Validation ──────────────────────────────────────────────────

def cross_validate(X_train, y_train, X_test, y_test, class_names,
                   n_splits=10, epochs=50, batch_size=32):
    """
    Train with 10-fold cross validation on dementia/control data,
    then evaluate on Alzheimer's/control test set.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_pred = [], []

    # Accumulators for averaged epoch-level metrics
    train_acc_accum  = np.zeros(epochs)
    val_acc_accum    = np.zeros(epochs)
    train_loss_accum = np.zeros(epochs)
    val_loss_accum   = np.zeros(epochs)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_fold_train, X_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_val = y_train[train_idx], y_train[val_idx]

        # Z-score normalisation
        X_fold_train     = z_score_normalize(X_fold_train)
        X_val            = z_score_normalize(X_val)
        X_test_norm      = z_score_normalize(X_test)

        model = build_model(input_shape=X_fold_train.shape[1:])
        history = model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Accumulate metrics
        train_acc_accum  += np.array(history.history['accuracy'])
        val_acc_accum    += np.array(history.history['val_accuracy'])
        train_loss_accum += np.array(history.history['loss'])
        val_loss_accum   += np.array(history.history['val_loss'])

        # Predict on Alzheimer's test set
        y_pred = (model.predict(X_test_norm) > 0.5).astype(int).flatten()
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Average metrics over all folds
    train_acc_accum  /= n_splits
    val_acc_accum    /= n_splits
    train_loss_accum /= n_splits
    val_loss_accum   /= n_splits

    print("\nClassification Report (Test Set — Alzheimer's vs Control):")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    plot_confusion_matrix(all_y_true, all_y_pred, class_names)
    plot_combined_metrics(train_acc_accum, val_acc_accum,
                          train_loss_accum, val_loss_accum)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir   = '/home/vidur/aarushi/datasetModel 2'
    train_path = os.path.join(base_dir, 'train')   # Contains: dementia/, control/
    test_path  = os.path.join(base_dir, 'test')    # Contains: alzheimers/, control/

    print("Loading training data (Dementia + Control)...")
    X_train, y_train, train_class_names = load_images_from_directory(train_path)

    print("Loading test data (Alzheimer's + Control)...")
    X_test, y_test, test_class_names = load_images_from_directory(test_path)

    print(f"\nTrain classes : {train_class_names}")
    print(f"Test  classes : {test_class_names}")
    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}")

    cross_validate(X_train, y_train, X_test, y_test,
                   class_names=test_class_names)
