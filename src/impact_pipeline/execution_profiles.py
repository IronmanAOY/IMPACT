from __future__ import annotations

from dataclasses import dataclass, field


HUNTER_SCHEDULERS = ("pbs", "slurm")


@dataclass(frozen=True)
class HunterSlurmProfile:
    """Generic Slurm backend (optional; HLRS Hunter itself runs PBS Pro)."""

    apu_partition: str = "apu"
    cpu_partition: str = "cpu"
    account: str | None = None
    qos: str | None = None
    phase1_time: str = "24:00:00"
    cut_time: str = "24:00:00"
    reduce_time: str = "02:00:00"
    cpus_per_task: int = 32
    mem_per_task: str = "0"
    gpus_per_task: int = 0
    # Slurm's default MaxArraySize is 1001 (largest index 1000).
    max_array_size: int = 1001
    array_throttle: int | None = None


@dataclass(frozen=True)
class HunterPBSProfile:
    """
    Altair PBS Pro backend for HLRS Hunter (kb.hlrs.de Batch_System_PBSPro_(Hunter)).

    A mi300a node (4x MI300A APU, 96 cores, 512 GB HBM) is allocated exclusively
    and charged in full, so shard jobs pack ``shards_per_node`` shards per node
    with PALS ``mpiexec`` (one rank per APU by default).
    """

    node_type: str = "mi300a"
    queue: str | None = None
    group_list: str | None = None
    workspace_resource: str | None = "ws13=True"
    localscratch: bool = False
    # Optional CPU/pre queue for the pure-CPU reduce/finalize job. Academic
    # users cannot use genoa nodes (routed to the restricted 'exception' queue),
    # so the default keeps reduce/finalize on mi300a.
    cpu_queue: str | None = None
    cpu_node_type: str | None = None
    phase1_time: str = "24:00:00"
    cut_time: str = "24:00:00"
    reduce_time: str = "04:00:00"
    max_walltime: str = "24:00:00"
    test_queue_max_walltime: str = "00:25:00"
    shards_per_node: int = 4
    cores_per_node: int = 96
    gpus_per_node: int = 4
    cpu_bind: str | None = None
    gpu_bind: str | None = None
    launcher: str = "mpiexec"
    # PBS Pro server default max_array_size (Hunter value unverified).
    max_array_size: int = 10000


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    description: str
    distributed_iim: bool
    hunter_phase1_shards_per_run: int = 1
    hunter_cut_shards_per_run: int = 1
    hunter_phase1_workers_per_task: int | None = None
    hunter_phase1_chunk_size: int = 8
    hunter_shared_memory: bool = True
    hunter_slurm: HunterSlurmProfile | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    hunter_scheduler: str = "pbs"
    hunter_pbs: HunterPBSProfile | None = None


LOCAL_EXECUTION_PROFILE = ExecutionProfile(
    name="local",
    description="Single-machine execution for workstation development and stepwise runs.",
    distributed_iim=False,
    hunter_phase1_shards_per_run=1,
    hunter_cut_shards_per_run=1,
    hunter_phase1_workers_per_task=None,
    hunter_phase1_chunk_size=8,
    hunter_shared_memory=True,
    hunter_slurm=None,
    notes=(
        "Uses the in-process pipeline and local multiprocessing only.",
        "Intended for MacBook-scale debugging, smoke tests, and individual step execution.",
    ),
)


HUNTER_EXECUTION_PROFILE = ExecutionProfile(
    name="hunter",
    description=(
        "Hunter-oriented distributed IIM execution with shared math and PBS Pro "
        "(default) or Slurm shard orchestration."
    ),
    distributed_iim=True,
    hunter_phase1_shards_per_run=16,
    hunter_cut_shards_per_run=256,
    # None = derive from the node packing at campaign build time
    # (PBS: cores_per_node / shards_per_node - 2; Slurm: cpus_per_task).
    hunter_phase1_workers_per_task=None,
    hunter_phase1_chunk_size=8,
    hunter_shared_memory=True,
    hunter_slurm=HunterSlurmProfile(),
    notes=(
        "IIM is decomposed into prepare, phase-1, cut-shard, and reduce stages.",
        "Non-IIM pipeline stages remain shared with local mode to avoid double maintenance.",
        "HLRS Hunter runs PBS Pro; PBS is the default backend. Slurm is optional.",
        "Shard, worker, packing and scheduler settings can be overridden via CLI "
        "flags or IMPACT_HUNTER_* variables.",
    ),
    hunter_scheduler="pbs",
    hunter_pbs=HunterPBSProfile(),
)


def normalize_hunter_scheduler(name: str | None, default: str = "pbs") -> str:
    key = str(name or default).strip().lower()
    aliases = {"pbspro": "pbs", "pbs-pro": "pbs", "openpbs": "pbs", "sbatch": "slurm"}
    key = aliases.get(key, key)
    if key not in HUNTER_SCHEDULERS:
        raise ValueError(
            f"Unknown Hunter scheduler '{name}'. "
            f"Expected one of: {', '.join(HUNTER_SCHEDULERS)}."
        )
    return key


def get_execution_profile(name: str) -> ExecutionProfile:
    key = str(name).strip().lower()
    if key == "local":
        return LOCAL_EXECUTION_PROFILE
    if key == "hunter":
        return HUNTER_EXECUTION_PROFILE
    raise ValueError(f"Unknown execution mode '{name}'. Expected one of: local, hunter.")
