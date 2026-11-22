"""Tuner & Trainer module untuk klasifikasi diabetes.

Modul ini dipakai oleh komponen Tuner (fungsi ``tuner_fn``) dan Trainer
(fungsi ``run_fn``) agar arsitektur model dan input pipeline konsisten.
"""

import functools
import os
from typing import Any, Dict, NamedTuple, Text

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "Outcome"
MONITOR_METRIC = "val_binary_accuracy"

TunerFnResult = NamedTuple(
    "TunerFnResult",
    [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])],
)


def transformed_name(key):
    """Menambahkan akhiran _xf pada nama fitur hasil transformasi."""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Membaca TFRecord terkompresi GZIP."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    """Menghasilkan tf.data.Dataset berisi pasangan (fitur, label) hasil transformasi.

    Args:
        file_pattern: pola path berkas TFRecord.
        tf_transform_output: objek TFTransformOutput dari komponen Transform.
        num_epochs: jumlah perulangan dataset (None berarti tak terbatas).
        batch_size: ukuran batch.

    Returns:
        tf.data.Dataset yang siap dipakai untuk pelatihan.
    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )


def get_feature_keys(tf_transform_output):
    """Mengambil nama fitur hasil transformasi (tanpa label) dari Transform."""
    label = transformed_name(LABEL_KEY)
    return sorted(
        key for key in tf_transform_output.transformed_feature_spec() if key != label
    )


def model_builder(hparams, feature_keys):
    """Membangun model Keras berdasarkan hyperparameter.

    Args:
        hparams: objek keras_tuner.HyperParameters.
        feature_keys: daftar nama fitur hasil transformasi.

    Returns:
        model tf.keras yang sudah di-compile.
    """
    inputs = [
        tf.keras.Input(shape=(1,), name=key, dtype=tf.float32) for key in feature_keys
    ]
    layer = tf.keras.layers.concatenate(inputs)

    dropout_rate = hparams.Float("dropout", 0.0, 0.5, step=0.1)
    for i in range(hparams.Int("num_hidden_layers", 1, 3)):
        layer = tf.keras.layers.Dense(
            hparams.Int(f"units_{i}", min_value=16, max_value=128, step=16),
            activation="relu",
        )(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hparams.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat fungsi serving yang menerima tf.Example terserialisasi."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Membangun tuner RandomSearch beserta argumen untuk proses fit."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output)

    tuner = kt.RandomSearch(
        hypermodel=functools.partial(
            model_builder, feature_keys=get_feature_keys(tf_transform_output)
        ),
        objective=kt.Objective(MONITOR_METRIC, direction="max"),
        max_trials=10,
        directory=fn_args.working_dir,
        project_name="diabetes_kt",
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=MONITOR_METRIC,
        mode="max",
        patience=5,
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 30,
            "callbacks": [early_stopping],
        },
    )


def run_fn(fn_args: FnArgs):
    """Melatih model dengan hyperparameter terbaik hasil Tuner lalu menyimpannya."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_set = input_fn(fn_args.train_files, tf_transform_output)
    eval_set = input_fn(fn_args.eval_files, tf_transform_output)

    hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    model = model_builder(hparams, get_feature_keys(tf_transform_output))
    model.summary()

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="batch"),
        tf.keras.callbacks.EarlyStopping(
            monitor=MONITOR_METRIC,
            mode="max",
            patience=10,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        x=train_set,
        validation_data=eval_set,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=50,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
