"""Membangun objek pipeline TFX yang dijalankan dengan Apache Beam."""

from typing import List, Text

from absl import logging
from tfx.orchestration import metadata, pipeline


def init_local_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    metadata_path: Text,
    components: List,
) -> pipeline.Pipeline:
    """Membuat pipeline TFX untuk dijalankan oleh BeamDagRunner.

    Args:
        pipeline_name: nama pipeline.
        pipeline_root: folder penyimpanan artefak pipeline.
        metadata_path: path database SQLite untuk ML Metadata.
        components: daftar komponen TFX.

    Returns:
        objek tfx.orchestration.pipeline.Pipeline.
    """
    logging.info("Pipeline root set to: %s", pipeline_root)
    beam_args = [
        "--direct_running_mode=in_memory",
        "--direct_num_workers=1",
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args,
    )
