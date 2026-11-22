"""Transform module untuk dataset Pima Indians Diabetes."""

import tensorflow as tf
import tensorflow_transform as tft

NUMERIC_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Kolom yang nilai 0-nya berarti data hilang (secara medis tidak mungkin 0).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

LABEL_KEY = "Outcome"


def transformed_name(key):
    """Menambahkan akhiran _xf pada nama fitur hasil transformasi."""
    return key + "_xf"


def _fill_zero_with_mean(values):
    """Mengganti nilai 0 (data hilang) dengan rata-rata nilai yang bukan 0."""
    is_valid = tf.not_equal(values, 0.0)
    valid_sum = tft.sum(tf.where(is_valid, values, tf.zeros_like(values)))
    valid_count = tft.sum(tf.cast(is_valid, tf.float32))
    mean_valid = valid_sum / tf.maximum(valid_count, 1.0)
    return tf.where(is_valid, values, tf.fill(tf.shape(values), mean_valid))


def preprocessing_fn(inputs):
    """Preprocessing fitur mentah menjadi fitur siap latih.

    Args:
        inputs: map dari nama fitur ke tensor mentah.

    Returns:
        map dari nama fitur hasil transformasi ke tensor.
    """
    outputs = {}

    for key in NUMERIC_FEATURES:
        values = tf.cast(inputs[key], tf.float32)
        if key in ZERO_AS_MISSING:
            values = _fill_zero_with_mean(values)
        outputs[transformed_name(key)] = tft.scale_to_z_score(values)

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
