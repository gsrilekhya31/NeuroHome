import zipfile
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


TRANSFORMER_PATH = Path(r"D:\dataset\models\p300_cnn_transformer.keras")
MEAN_PATH = Path(r"D:\dataset\models\cnn_transformer_mean.npy")
STD_PATH = Path(r"D:\dataset\models\cnn_transformer_std.npy")


def transformer_model():
    inputs = layers.Input(shape=(32, 256), name="eeg_input")

    x = layers.Permute((2, 1), name="permute")(inputs)
    x = layers.Reshape((256, 32, 1), name="reshape_input")(x)

    x = layers.Conv2D(
        16, (1, 15), padding="same",
        name="temporal_conv1"
    )(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("elu", name="elu1")(x)

    x = layers.Conv2D(
        32, (1, 7), padding="same",
        name="temporal_conv2"
    )(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("elu", name="elu2")(x)

    x = layers.Conv2D(
        32, (1, 1), padding="same",
        name="spatial_projection"
    )(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("elu", name="elu3")(x)

    x = layers.AveragePooling2D(
        pool_size=(1, 2),
        name="average_pool"
    )(x)

    x = layers.Reshape(
        (128, 32),
        name="temporal_tokens"
    )(x)

    x = layers.Dense(
        64,
        name="embedding"
    )(x)

    positions = tf.range(
        start=0,
        limit=128,
        delta=1
    )

    position_embedding = layers.Embedding(
        input_dim=128,
        output_dim=64,
        name="position_embedding"
    )(positions)

    x = x + position_embedding

    for block in range(3):
        prefix = f"transformer_{block + 1}"

        attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=16,
            dropout=0.2,
            name=f"{prefix}_mha"
        )(x, x)

        x = layers.Add(
            name=f"{prefix}_attention_add"
        )([x, attention])

        x = layers.LayerNormalization(
            name=f"{prefix}_attention_norm"
        )(x)

        ffn = layers.Dense(
            128,
            activation="gelu",
            name=f"{prefix}_ffn1"
        )(x)

        ffn = layers.Dropout(
            0.2,
            name=f"{prefix}_dropout"
        )(ffn)

        ffn = layers.Dense(
            64,
            name=f"{prefix}_ffn2"
        )(ffn)

        x = layers.Add(
            name=f"{prefix}_ffn_add"
        )([x, ffn])

        x = layers.LayerNormalization(
            name=f"{prefix}_ffn_norm"
        )(x)

    attention_scores = layers.Dense(
        1,
        name="attention_scores"
    )(x)

    attention_weights = layers.Softmax(
        axis=1,
        name="attention_weights"
    )(attention_scores)

    x = layers.Multiply(
        name="attention_weighted"
    )([x, attention_weights])

    x = layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1),
        output_shape=(64,),
        name="attention_pool"
    )(x)

    x = layers.Dense(
        64,
        activation="gelu",
        name="classifier_dense"
    )(x)

    x = layers.Dropout(
        0.30,
        name="classifier_dropout"
    )(x)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="output"
    )(x)

    return Model(inputs, outputs, name="CNN_Transformer")


def load_transformer():
    model = transformer_model()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        with zipfile.ZipFile(
            TRANSFORMER_PATH,
            "r"
        ) as z:
            weights_name = "model.weights.h5"

            if weights_name not in z.namelist():
                raise RuntimeError(
                    "model.weights.h5 not found inside Transformer .keras file"
                )

            weights_path = td / weights_name
            weights_path.write_bytes(
                z.read(weights_name)
            )

        model.load_weights(weights_path)

    mean = np.load(MEAN_PATH)
    std = np.load(STD_PATH)

    return model, mean.astype(np.float32), std.astype(np.float32)
