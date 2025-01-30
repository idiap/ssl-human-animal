# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from bob.extension import rc
from dask.distributed import Client
from dask_jobqueue import SGECluster
from hydra.core.config_store import ConfigStore
from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    configure_log,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

# IMPORTANT:
# If your plugin imports any module that takes more than a fraction of a second to import,
# Import the module lazily (typically inside launch()).
# Installed plugins are imported during Hydra initialization and plugins that are slow to import plugins will slow
# the startup of ALL hydra applications.
# Another approach is to place heavy includes in a file prefixed by _, such as _core.py:
# Hydra will not look for plugin in such files and will not import them during plugin discovery.


log = logging.getLogger(__name__)


@dataclass
class LauncherConfig:
    _target_: str = "hydra_plugins.hydra_dask_launcher.dask_launcher.DaskLauncher"
    queue: Optional[str] = None
    project: str = "evolang"
    cpumem: str = "8GB"
    gpumem: Optional[str] = None
    walltime: str = "03:00:00"
    log_directory: Optional[str] = None
    hostname: Optional[str] = None


ConfigStore.instance().store(group="hydra/launcher", name="dask", node=LauncherConfig)


class DaskLauncher(Launcher):
    def __init__(
        self,
        queue: str,
        project: str,
        cpumem: str,
        gpumem: str,
        walltime: str,
        log_directory: str,
        hostname: str,
    ) -> None:
        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

        # foo and var are coming from the the plugin's configuration
        self.queue = queue
        self.project = project
        self.cpumem = cpumem
        self.gpumem = gpumem
        self.walltime = walltime
        self.log_directory = log_directory
        self.hostname = hostname

    def get_resource_spec(self):
        resource_specs = []
        if self.queue is not None:
            resource_specs.append(f"{self.queue}")
        if self.gpumem is not None:
            resource_specs.append(f"gpumem={self.gpumem}")
        if self.hostname is not None:
            resource_specs.append(f"hostname={self.hostname}")

        if resource_specs:
            return ",".join(resource_specs)

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

        self.cluster = SGECluster(
            # queue=self.queue, # Needed else jobs can run for longer ?
            project=self.project,
            cores=1,
            memory=self.cpumem,
            resource_spec=self.get_resource_spec(),
            nanny=False,
            walltime=self.walltime,
            job_script_prologue=[
                f"export PROJECT_ROOT={os.environ['PROJECT_ROOT']}",
                f"export TEMP_RESULTS_ROOT={os.environ['TEMP_RESULTS_ROOT']}",
                f"export USER_RESULTS_ROOT={os.environ['USER_RESULTS_ROOT']}",
            ],
            log_directory=self.log_directory,
        )

        self.client = Client(self.cluster)

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """
        :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
        :param initial_job_idx: Initial job idx in batch.
        :return: an array of return values from run_job with indexes corresponding to the input list indexes.
        """
        setup_globals()
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Dask Launcher is launching {len(job_overrides)} jobs")
        log.info(f"Sweep output dir : {sweep_dir}")
        log.info(f"Cluster configuration: {self.cluster.job_header}")
        runs = []

        n_jobs = len(job_overrides)

        self.client.restart(wait_for_workers=False)
        self.cluster.adapt(minimum_jobs=0, maximum_jobs=n_jobs)

        state = Singleton.get_state()

        def run_job_with_singleton(run_job, state, **kwargs):
            Singleton.set_state(state)
            return run_job(**kwargs)

        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {lst}")
            sweep_config = self.hydra_context.config_loader.load_sweep_config(
                self.config, list(overrides)
            )

            with open_dict(sweep_config):
                # This typically coming from the underlying scheduler (SLURM_JOB_ID for instance)
                # In that case, it will not be available here because we are still in the main process.
                # but instead should be populated remotely before calling the task_function.
                sweep_config.hydra.job.id = f"job_id_for_{idx}"
                sweep_config.hydra.job.num = idx

            future = self.client.submit(
                run_job_with_singleton,
                state=state,
                run_job=run_job,
                hydra_context=self.hydra_context,
                task_function=self.task_function,
                config=sweep_config,
                job_dir_key="hydra.sweep.dir",
                job_subdir_key="hydra.sweep.subdir",
            )
            runs.append(future)
            # reconfigure the logging subsystem for Hydra as the run_job call configured it for the Job.
            # This is needed for launchers that calls run_job in the same process and not spawn a new one.
            configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)

        return self.client.gather(runs)
