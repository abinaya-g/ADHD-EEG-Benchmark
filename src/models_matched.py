"""
Matched-head control for the CNN+LR comparison.

CNN+LR differs from the plain CNN in several ways at once: the classifier is a
separate L2-penalised logistic regression fitted to convergence on the fixed
64-unit features, while the plain CNN's 32-unit dense layer and sigmoid output
are trained jointly with early stopping. This control keeps the CNN trunk
identical (everything up to and including the 64-unit dense layer) and replaces
the 32-unit dense layer + sigmoid by a single regularised linear sigmoid unit
on the 64-d features, trained jointly under exactly the same optimizer, batch
size, early-stopping rule and nested subject-grouped protocol. If this model
recovers the CNN+LR advantage, the gain is attributable to the form of the
classification head and its regularisation rather than to a separately fitted
"hybrid" classifier.

L2 strength 1e-3 on the head weights corresponds roughly to scikit-learn's
C = 1 (default) for ~400 training epochs (1 / (2 * C * n) ~ 1.2e-3).
"""
from tensorflow.keras import layers, models, regularizers


def build_cnn_linear_head(n_channels=19, n_timesamples=3840, fs=128, l2=1e-3):
    temporal_kernel_1 = int(fs / 4)
    temporal_kernel_2 = int(fs / 8)
    inp = layers.Input(shape=(n_channels, n_timesamples, 1), name="input_eeg")
    x = layers.Conv2D(16, (10, 1), padding="valid", activation="relu", name="spatial_conv1")(inp)
    x = layers.BatchNormalization(name="bn_spatial1")(x)
    x = layers.AveragePooling2D((2, 1), name="pool_spatial1")(x)
    x = layers.Conv2D(16, (4, 1), padding="valid", activation="relu", name="spatial_conv2")(x)
    x = layers.BatchNormalization(name="bn_spatial2")(x)
    x = layers.AveragePooling2D((2, 1), name="pool_spatial2")(x)
    x = layers.Reshape((x.shape[2], x.shape[3]), name="reshape_to_temporal")(x)
    x = layers.Conv1D(32, temporal_kernel_1, padding="valid", activation="relu", name="temporal_conv1")(x)
    x = layers.BatchNormalization(name="bn_temporal1")(x)
    x = layers.AveragePooling1D(temporal_kernel_1 // 2, name="pool_temporal1")(x)
    x = layers.Conv1D(32, temporal_kernel_2, padding="valid", activation="relu", name="temporal_conv2")(x)
    x = layers.BatchNormalization(name="bn_temporal2")(x)
    x = layers.AveragePooling1D(temporal_kernel_2 // 2, name="pool_temporal2")(x)
    flat = layers.Flatten(name="flatten")(x)
    dense1 = layers.Dense(64, activation="relu", name="dense1")(flat)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(l2), name="classification")(dense1)
    model = models.Model(inputs=inp, outputs=out, name="adhd_cnn_linear_head")
    feature_model = models.Model(inputs=inp, outputs=dense1, name="adhd_cnn_linear_head_features")
    return model, feature_model
