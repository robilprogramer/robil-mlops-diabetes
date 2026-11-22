"""Menjalankan pipeline TFX klasifikasi diabetes dengan Apache Beam (BeamDagRunner)."""

import os

from absl import logging
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from modules.components import ComponentConfig, init_components
from modules.pipeline import init_local_pipeline

PIPELINE_NAME = "robil-pipeline"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT_DIR, "data")
TRANSFORM_MODULE_FILE = os.path.join(ROOT_DIR, "modules", "diabetes_transform.py")
TRAINER_MODULE_FILE = os.path.join(ROOT_DIR, "modules", "diabetes_trainer.py")

PIPELINE_ROOT = os.path.join(ROOT_DIR, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
SERVING_MODEL_DIR = os.path.join(ROOT_DIR, "serving_model_dir")


def main():
    """Membuat komponen, menyusun pipeline, lalu menjalankannya dengan Beam."""
    logging.set_verbosity(logging.INFO)

    components = init_components(
        ComponentConfig(
            data_dir=DATA_ROOT,
            transform_module=TRANSFORM_MODULE_FILE,
            training_module=TRAINER_MODULE_FILE,
            training_steps=500,
            eval_steps=100,
            tuning_steps=100,
            serving_model_dir=SERVING_MODEL_DIR,
        )
    )

    BeamDagRunner().run(
        init_local_pipeline(PIPELINE_NAME, PIPELINE_ROOT, METADATA_PATH, components)
    )


if __name__ == "__main__":
    main()
