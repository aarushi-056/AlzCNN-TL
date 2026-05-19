import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os


# ── Preprocessing ─────────────────────────────────────────────────────────────

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
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = load_img(os.path.join(class_dir, filename),
                               target_size=target_size)
                img_array = img_to_array(img)
                img_array = skull_strip(img_array)
                images.append(img_array)
                labels.append(class_map[class_name])

    return np.array(images), np.array(labels), class_names


# ── Model Definitions ─────────────────────────────────────────────────────────

def build_bob_model(input_shape=(128, 128, 3), num_classes=5):
    """
    Bob: 5-class CNN pretrained on MCI data.
    Architecture:
      Input →
      Conv Block 1: Conv(32) → Pool → Dropout(0.25) →
      Conv Block 2: Conv(64) → Pool → Dropout(0.25) →
      Conv Block 3: Conv(64) → Pool → Dropout(0.25) →
      Flatten → Dense(256) → Dropout(0.25) → Softmax(5)
    Classes: EMCI, LMCI, MCI, Control, Alzheimer's
    Loss: categorical cross-entropy
    """
    model = Sequential([
        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_rob_model(bob_model, input_shape=(128, 128, 3)):
    """
    Rob: Binary classifier built on top of frozen Bob feature extractor.
    Architecture:
      Input → Frozen Bob layers (feature extraction) →
      Dense(128) → Dropout(0.25) → Sigmoid output
    Loss: binary cross-entropy
    """
    for layer in bob_model.layers:
        layer.trainable = False

    inputs = Input(shape=input_shape)
    x = bob_model(inputs, training=False)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Average Confusion Matrix — Model 5')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved as '{filename}'.")


def plot_combined_metrics(train_acc, val_acc, train_loss, val_loss, filename):
    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc,  label='Train Accuracy')
    plt.plot(epochs, val_acc,    label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Rob Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss,   label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Rob Model Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Metrics plot saved as '{filename}'.")


# ── Stage 1: Train Bob on 5-Class MCI Data ────────────────────────────────────

def train_bob(X_train, y_train, X_tune, y_tune,
              n_splits=10, epochs=50, batch_size=32, num_classes=5):
    """
    Train Bob (5-class) with 10-fold cross validation on MCI data.
    Returns the best Bob model (lowest validation loss).
    Dataset split: 70% train / 15% tune / 15% test (per paper, Table 8).
    Classes: EMCI(336), LMCI(100), MCI(808), Control(1596), Alzheimer's(350)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_bob      = None
    best_val_loss = float('inf')

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n[Bob] Fold {fold + 1}/{n_splits}")

        X_fold_train = z_score_normalize(X_train[train_idx])
        X_val        = z_score_normalize(X_train[val_idx])
        y_fold_train = to_categorical(y_train[train_idx], num_classes)
        y_val_cat    = to_categorical(y_train[val_idx],   num_classes)

        bob = build_bob_model(input_shape=X_fold_train.shape[1:],
                              num_classes=num_classes)
        history = bob.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        final_val_loss = history.history['val_loss'][-1]
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_bob = bob
            print(f"  → New best Bob (val_loss={best_val_loss:.4f})")

    print("\nBob training complete. Best val_loss: {:.4f}".format(best_val_loss))
    return best_bob


# ── Stage 2: Train Rob (Binary) Using Bob's Frozen Features ───────────────────

def train_rob(bob_model, X_train, y_train, X_test, y_test, class_names,
              n_splits=10, epochs=50, batch_size=32):
    """
    Train Rob (binary) with 10-fold cross validation.
    Bob's layers are frozen and used as a feature extractor.
    Evaluated on Alzheimer's/control test set.
    Dataset: 500 AD + 4060 non-AD (train) | 200 AD + 200 Control (test)
    per paper, Table 9.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_pred = [], []

    train_acc_accum  = np.zeros(epochs)
    val_acc_accum    = np.zeros(epochs)
    train_loss_accum = np.zeros(epochs)
    val_loss_accum   = np.zeros(epochs)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n[Rob] Fold {fold + 1}/{n_splits}")

        X_fold_train = z_score_normalize(X_train[train_idx])
        X_val        = z_score_normalize(X_train[val_idx])
        X_test_norm  = z_score_normalize(X_test)
        y_fold_train = y_train[train_idx]
        y_val        = y_train[val_idx]

        rob = build_rob_model(bob_model, input_shape=X_fold_train.shape[1:])
        history = rob.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        train_acc_accum  += np.array(history.history['accuracy'])
        val_acc_accum    += np.array(history.history['val_accuracy'])
        train_loss_accum += np.array(history.history['loss'])
        val_loss_accum   += np.array(history.history['val_loss'])

        y_pred = (rob.predict(X_test_norm) > 0.5).astype(int).flatten()
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    # Average over folds
    train_acc_accum  /= n_splits
    val_acc_accum    /= n_splits
    train_loss_accum /= n_splits
    val_loss_accum   /= n_splits

    print("\nClassification Report (Test Set — Alzheimer's vs Control):")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    plot_confusion_matrix(all_y_true, all_y_pred, class_names,
                          'confusion_matrix_model5.png')
    plot_combined_metrics(train_acc_accum, val_acc_accum,
                          train_loss_accum, val_loss_accum,
                          'training_validation_metrics_model5.png')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = '/home/vidur/aarushi/datasetModel5'

    # ── Stage 1 data: 5-class MCI (train + tune subfolders) ───────────────────
    # Classes: EMCI, LMCI, MCI, Control, Alzheimer's
    # Split: 70% train / 15% tune / 15% test (per paper, Table 8)
    bob_train_path = os.path.join(base_dir, 'train')   # 70%
    bob_tune_path  = os.path.join(base_dir, 'tune')    # 15%

    print("Loading Bob training data (5-class MCI)...")
    X_bob_train, y_bob_train, bob_class_names = load_images_from_directory(bob_train_path)
    X_bob_tune,  y_bob_tune,  _               = load_images_from_directory(bob_tune_path)

    print(f"Bob classes  : {bob_class_names}")
    print(f"Bob train    : {len(X_bob_train)} samples")
    print(f"Bob tune     : {len(X_bob_tune)}  samples")

    # ── Stage 2 data: binary AD vs Control ────────────────────────────────────
    # Train: 500 AD + 4060 non-AD  |  Test: 200 AD + 200 Control (per paper, Table 9)
    rob_train_path = os.path.join(base_dir, 'rob_train')
    rob_test_path  = os.path.join(base_dir, 'rob_test')

    print("\nLoading Rob training data (AD + non-AD)...")
    X_rob_train, y_rob_train, _               = load_images_from_directory(rob_train_path)

    print("Loading Rob test data (Alzheimer's + Control)...")
    X_rob_test,  y_rob_test,  rob_class_names = load_images_from_directory(rob_test_path)

    print(f"Rob classes  : {rob_class_names}")
    print(f"Rob train    : {len(X_rob_train)} samples")
    print(f"Rob test     : {len(X_rob_test)}  samples")

    # ── Train Bob ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1: Training Bob (5-class MCI classifier)")
    print("="*60)
    best_bob = train_bob(X_bob_train, y_bob_train,
                         X_bob_tune,  y_bob_tune,
                         num_classes=len(bob_class_names))

    # ── Train Rob ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: Training Rob (binary AD classifier via transfer learning)")
    print("="*60)
    train_rob(best_bob,
              X_rob_train, y_rob_train,
              X_rob_test,  y_rob_test,
              class_names=rob_class_names)
