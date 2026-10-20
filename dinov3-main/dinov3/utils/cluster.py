# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class ClusterType(Enum):
    DEFAULT = "cw"


def _guess_cluster_type() -> ClusterType:
    return ClusterType.DEFAULT


def get_cluster_type(
    cluster_type: Optional[ClusterType] = None,
) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_slurm_account(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    return os.environ.get("SLURM_ACCOUNT")

def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    root = os.environ.get("DINOV3_CHECKPOINT_ROOT")
    return Path(root).expanduser() if root else None

def get_user_checkpoint_path(
    cluster_type: Optional[ClusterType] = None,
) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None

    username = os.environ.get("USER")
    assert username is not None
    return checkpoint_path / username


def get_slurm_qos(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    return os.environ.get("SLURM_QOS")

def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    return os.environ.get("SLURM_PARTITION")

def get_slurm_executor_parameters(
    nodes: int,
    num_gpus_per_node: int,
    cluster_type: Optional[ClusterType] = None,
    **kwargs,
) -> Dict[str, Any]:
    # create default parameters
    params = {
        "mem_gb": 0,  # Requests all memory on a node, see https://slurm.schedmd.com/sbatch.html
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "cpus_per_task": 10,
        "nodes": nodes,
        "slurm_partition": get_slurm_partition(cluster_type),
    }
    # apply cluster-specific adjustments
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type == ClusterType.DEFAULT:
        params["cpus_per_task"] = 16
    # set additional parameters / apply overrides
    params.update(kwargs)
    return params
