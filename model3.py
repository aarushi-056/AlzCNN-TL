# MODEL 3
# 5-Class CNN + Transfer Learning Binary Classifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)
from tensorflow.keras.optimizers import Adam


# =========================================================
# BOB MODEL
# 5-Class Classification CNN
# =========================================================

def create_bob_model(
    input_shape=(128, 128, 3),
    num_classes=5,
    learning_rate=0.001
):

    model = Sequential([

        Input(shape=input_shape),

        # Block 1
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Flatten(),

        Dense(256, activation='relu'),
        Dropout(0.25),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# =========================================================
# ROB MODEL
# Transfer Learning Binary Classifier
# =========================================================

def create_rob_model(
    bob_model,
    input_shape=(128,128,3),
    learning_rate=0.001
):

    # Freeze Bob layers
    for layer in bob_model.layers:
        layer.trainable = False

    inputs = Input(shape=input_shape)

    x = bob_model(inputs, training=False)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    rob_model = Model(inputs, outputs)

    rob_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return rob_model



bob_model = create_bob_model()

rob_model = create_rob_model(bob_model)

bob_model.summary()
rob_model.summary()
