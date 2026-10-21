"""CNN backbones + shared classification head (Sec. 3.5-3.8).

Head: GAP -> Dense(512, ReLU) -> Dropout(0.5) -> Dense(256, ReLU) -> Dropout(0.3) -> Dense(5, Softmax)
"""

from __future__ import annotations

import keras

from common import config

BACKBONES = {
    "ResNet50": keras.applications.ResNet50,
    "InceptionV3": keras.applications.InceptionV3,
    "Xception": keras.applications.Xception,
}


def build_model(
    model_name: str,
    pretrained: bool,
    num_classes: int = config.NUM_CLASSES,
    fine_tune_last_n: int = config.FINE_TUNE_LAST_N_LAYERS,
) -> keras.Model:
    """Build a classifier.

    pretrained=True  -> ImageNet weights, only the last `fine_tune_last_n` backbone
                        layers are trainable (code_with_weights).
    pretrained=False -> random initialisation, every layer trainable
                        (code_without_weights, trained from scratch).
    """
    if model_name not in BACKBONES:
        raise ValueError(f"Unknown model {model_name}; choose from {list(BACKBONES)}")
    size = config.INPUT_SIZE[model_name]
    backbone = BACKBONES[model_name](
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(size, size, 3),
    )

    if pretrained:
        for layer in backbone.layers[:-fine_tune_last_n]:
            layer.trainable = False
        for layer in backbone.layers[-fine_tune_last_n:]:
            layer.trainable = True
    else:
        backbone.trainable = True

    inputs = keras.Input(shape=(size, size, 3))
    x = backbone(inputs, training=False if pretrained else None)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    for units, rate, i in zip(config.DENSE_UNITS, config.DROPOUT_RATES, range(1, 3)):
        x = keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = keras.layers.Dropout(rate, name=f"dropout_{i}")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    tag = "pretrained" if pretrained else "scratch"
    return keras.Model(inputs, outputs, name=f"{model_name}_{tag}")
