"""Inisialisasi seluruh komponen TFX untuk pipeline klasifikasi diabetes."""

from typing import NamedTuple

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

LABEL_KEY = "Outcome"


class ComponentConfig(NamedTuple):
    """Konfigurasi yang dibutuhkan untuk membuat komponen pipeline."""

    data_dir: str
    transform_module: str
    training_module: str
    training_steps: int
    eval_steps: int
    tuning_steps: int
    serving_model_dir: str


def _create_eval_config():
    """Membuat konfigurasi evaluasi TFMA beserta threshold blessing."""
    metric_names = [
        "ExampleCount",
        "AUC",
        "Precision",
        "Recall",
        "FalsePositives",
        "TruePositives",
        "FalseNegatives",
        "TrueNegatives",
    ]
    metrics = [tfma.MetricConfig(class_name=name) for name in metric_names]
    metrics.append(
        tfma.MetricConfig(
            class_name="BinaryAccuracy",
            threshold=tfma.MetricThreshold(
                value_threshold=tfma.GenericValueThreshold(lower_bound={"value": 0.6}),
                change_threshold=tfma.GenericChangeThreshold(
                    direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                    absolute={"value": 0.0001},
                ),
            ),
        )
    )

    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=LABEL_KEY)],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[tfma.MetricsSpec(metrics=metrics)],
    )


def init_components(config: ComponentConfig):
    """Membuat seluruh komponen TFX sesuai urutan eksekusi.

    Args:
        config: konfigurasi komponen (lihat ComponentConfig).

    Returns:
        tuple komponen TFX: ExampleGen, StatisticsGen, SchemaGen,
        ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator,
        dan Pusher.
    """
    example_gen = CsvExampleGen(
        input_base=config.data_dir,
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(
                splits=[
                    example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                    example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
                ]
            )
        ),
    )

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=config.transform_module,
    )

    tuner = Tuner(
        module_file=config.training_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=config.tuning_steps),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=config.eval_steps),
    )

    trainer = Trainer(
        module_file=config.training_module,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"], num_steps=config.training_steps
        ),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=config.eval_steps),
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=_create_eval_config(),
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=config.serving_model_dir
            )
        ),
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )
