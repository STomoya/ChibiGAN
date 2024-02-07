"""Training status logger."""

from __future__ import annotations

import atexit
import datetime
import pprint
import subprocess
import sys
import time
from argparse import ArgumentParser, Namespace
from collections import deque
from contextlib import contextmanager
from statistics import mean
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from src.utils import Path, get_logger


class Collector:
    """Collect values by summing values until .update() is called, then reset."""

    def __init__(self) -> None:
        """Construct object."""
        self._deltas = dict()

    @torch.no_grad()
    def report(self, name: str, value: float | torch.Tensor) -> None:
        """Report a value with an identical name to trace.

        Args:
        ----
            name (str): Key for the value. This name will be used at .add_scalar() of SummaryWriter.
            value (float | torch.Tensor): The value to collect.
        """
        if value is None:
            return

        if name not in self._deltas:
            self._deltas[name] = [0, 0]

        if torch.is_tensor(value):
            value = value.detach().cpu()
            num = value.numel()
            total = value.sum().item()
        else:
            num, total = 1, value

        self._deltas[name][0] += num
        self._deltas[name][1] += total

    def report_by_dict(self, step: dict[str, Any]) -> None:
        """Report values using a dict object. See .report() for details.

        Args:
        ----
            step (dict[str, Any]): dict of values to report.
        """
        for name, value in step.items():
            self.report(name, value)

    def mean(self, name: str) -> float:
        """Return the mean value of the collected value. If not exist or total is 0, returns inf.

        Args:
        ----
            name (str): Key used to report values.

        Returns:
        -------
            float: mean of the collected values.
        """
        if name not in self._deltas or self._deltas[name][0] == 0:
            return float("inf")
        return self._deltas[name][1] / self._deltas[name][0]

    def update(self) -> dict[str, float]:
        """Return mean of all collected values and reset.

        Returns
        -------
            dict[str, float]: dict of mean of all reported values.
        """
        output = {}
        for name in self._deltas:
            mean = self.mean(name)
            if mean != float("inf"):
                output[name] = self.mean(name)
            self._deltas[name] = [0, 0]
        return output


class Status:
    """Class for logging training status.

    Examples
    --------
        >>> status = Status(1000, './checkpoint/log.log')
        >>> while not status.is_end():
        >>>     output = model(input)
        >>>     # update training state. also updates the training progress.
        >>>     status.update(**{
        >>>         'Loss/CE/train': torch.rand(1),
        >>>         'Metrics/accuracy/train': torch.rand(1),
        >>>         'Scores/output': output, # tensors with any shapes can be passed.
        >>>         'Progress/lr': lr # float/int are also supported.
        >>>     })
        >>>     # update without updating training progress.
        >>>     status.dry_update(**{
        >>>         'Loss/CE/val': torch.rand(1),
        >>>         'Metrics/accuracy/val': torch.rand(1),
        >>>     })
    """

    def __init__(
        self,
        max_iters: int,
        log_file: str,
        log_interval: int = 1,
        logger_name: str = "logger",
        steptime_num_accum: int = 300,
        log_frequently_until: int = 100,
        log_nvidia_smi_at: int = 10,
        tb_folder: str | None = None,
        delta_format: str = "{key}: {value: 10.5f}",
    ) -> None:
        """Training status logger.

        Args:
        ----
            max_iters (int): Maximum iterations to train.
            log_file (str): Path to file for output logging to.
            bar (bool, optional): Enable tqdm progress bar. Default: False.
            log_interval (int, optional): Interval for logging status. Default: 1.
            logger_name (str, optional): The name of the logger. Default: 'logger'.
            steptime_num_accum (int, optional): Number of iterations to accumulate for calculating the rolling ETA.
                Default: 300.
            log_frequently_until (int): Do logging every 5 update until given int. Default: 100.
            log_nvidia_smi_at (int): Log `nvidia-smi` command at. Default: 10
            tb_folder (str | None, optional): Folder to save the tensorboard event.
                If not given, the parent folder of 'log_file' will be used. Default: None.
            delta_format (_type_, optional): The format used to print the collected values.
                - key: The name used to identify the value.
                - value: The value.
                Default: '{key}: {value: 10.5f}'.
        """
        self._max_iters = max_iters
        self._batches_done = 0
        self._log_file = log_file
        self._log_interval = log_interval
        self._log_frequently_until = log_frequently_until
        self._log_nvidia_smi_at = log_nvidia_smi_at

        log_file = Path(log_file)
        self._logger = get_logger(
            logger_name,
            filename=log_file,
            mode="a",
            format="%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s",
        )
        self._delta_format = delta_format

        self._collector = Collector()

        self._step_start = time.time()
        self._steptime_num_accum = steptime_num_accum
        self._steptimes = deque(maxlen=steptime_num_accum)

        self._tb_folder = (
            log_file.resolve().dirname() if tb_folder is None else tb_folder
        )
        self._tbwriter = SummaryWriter(self._tb_folder)
        self._profiler = None

        atexit.register(self._shutdown_logger)

    @property
    def max_iters(self):
        """Maximum iteration."""
        return self._max_iters

    @property
    def batches_done(self):
        """Current training progress."""
        return self._batches_done

    @batches_done.setter
    def batches_done(self, value):
        self._batches_done = value

    def get_kbatches(self, format="{kbatches:.2f}k") -> str:
        """Format `batches_done` to kilo batches.

        Args:
        ----
            format (str, optional): format of the string. Default: '{kbatches:.2f}k'.

        Returns:
        -------
            str: The formated kilo batches.
        """
        kbatches = self._batches_done / 1000
        return format.format(kbatches=kbatches)

    """print functions"""

    def log(self, message: str, level="info") -> None:
        """Log a message.

        Args:
        ----
            message (str): The message to log.
            level (str, optional): log level. Default: 'info'.
        """
        getattr(self._logger, level)(message)

    """Information loggers"""

    def log_command_line(self) -> None:
        """Log command line used to execute the python script."""
        command_line = sys.argv
        command_line = pprint.pformat(command_line)
        self.log(f"Execution command\n{command_line}")

    def log_args(
        self,
        args: Namespace,
        parser: ArgumentParser | None = None,
        filename: str | None = None,
    ) -> None:
        """Log argparse.Namespace obj.

        Args:
        ----
            args (Namespace): The command line arguments parsed by argparse.
            parser (ArgumentParser, optional): Parser used to parse the command line arguments.
                Used to display the default values. Default: None.
            filename (str, optional): A filename. if given the arguments will be saved to this file.
                Default: None.
        """
        message = "------------------------- Options -----------------------\n"
        for k, v in sorted(vars(args).items()):
            comment = ""
            if parser is not None and v != parser.get_default(k):
                comment = f"[default: {parser.get_default(k)}]"
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "------------------------- End ---------------------------"
        self.log(f"Command line arguments\n{message}")

        if filename is not None:
            with open(filename, "w") as fout:
                fout.write(message)
                fout.write("\n")

    def log_omegaconf(self, config: DictConfig) -> None:
        """Log omegaconf.DictConfig obj.

        Args:
        ----
            config (DictConfig): The config to log.
        """
        yamlconfig = OmegaConf.to_yaml(config)
        self.log(f"Config:\n{yamlconfig}")

    def log_dataset(self, dataloader: DataLoader) -> None:
        """Log DataLoader obj.

        Args:
        ----
            dataloader (DataLoader): The DataLoader object to log.
        """
        dataset = dataloader.dataset
        sampler = dataloader.sampler
        shuffle = False
        drop_last = False
        if isinstance(sampler, RandomSampler):
            shuffle = True
            drop_last = dataloader.drop_last
        elif isinstance(sampler, DistributedSampler):
            shuffle = sampler.shuffle
            drop_last = sampler.drop_last

        loader_kwargs = dict(
            TYPE=dataset.__class__.__name__,
            num_samples=len(dataset),
            num_iterations=len(dataloader),
            batch_size=dataloader.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
        )
        message = "------------------------- Dataset -----------------------\n"
        for k, v in sorted(loader_kwargs.items()):
            message += "{:>25}: {:<30}\n".format(str(k), str(v))
        message += "------------------------- End ---------------------------"
        self.log(f"Dataset\n{message}")

    def log_optimizer(self, optimizer: Optimizer) -> None:
        """Log optimizer obj.

        Args:
        ----
            optimizer (Optimizer): The optimizer object to log.
        """
        self.log(f"Optimizer:\n{optimizer}")

    def log_env(self) -> None:
        """Log pytorch build enviornment."""
        env = get_pretty_env_info()
        self.log(f"PyTorch environment:\n{env}")

    def log_model(self, model: torch.nn.Module) -> None:
        """Log nn.Module obj.

        Args:
        ----
            model (torch.nn.Module): The module to log.
        """
        self.log(f"Architecture: {model.__class__.__name__}:\n{model}")

    def log_gpu_memory(
        self,
        stage: str | None = None,
        at: list[int] | int | None = None,
        as_hook: bool = False,
    ) -> None:
        """Log memory summary.

        Optionally this function returns a executable hook that logs GPU memory when called.
        Useful for registering this function as a hook for `OptimizerStep`.

        Args:
        ----
            stage (str, optional): The name of the stage to summarize VRAM. Default: None.
            at (list[int], optional): Used to determine when to log summary.
                If None always log summary. Default: None.
            as_hook (bool, option): return a function that can be executed without arguments. Default: False.

        Example:
        -------
            >>> status = Status(...)
            >>> output = model(input)
            >>> status.log_gpu_memory('forward', [0, 100])
            >>> output.sum().backward()
            >>> status.log_gpu_memory('backward', [0, 100])
            >>> optimizer.step()
            >>> status.log_gpu_memory('step', [0, 100])
        """

        def hook():
            if torch.cuda.is_available():
                message = "GPU memory summary"
                if isinstance(stage, str):
                    message += f' of stage "{stage}"'
                message += f" at iteration {self.batches_done}."
                if (at is None) or (
                    self.batches_done == at
                    if isinstance(at, int)
                    else self.batches_done in at
                ):
                    message += f"\n{torch.cuda.memory_summary()}"
                    self.log(message)
            else:
                self.log("No GPU available on your enviornment.")

        if as_hook:
            return hook

        hook()

    def log_nvidia_smi(self) -> None:
        """Log nvidia-smi output."""
        if torch.cuda.is_available():
            nvidia_smi_output = subprocess.run(
                "nvidia-smi",
                shell=True,
                capture_output=True,
                universal_newlines=True,
                check=False,
            )
            self.log(f"\n{nvidia_smi_output.stdout}")
        else:
            self.log("No GPU available on your enviornment.")

    def log_actual_batch_size(
        self,
        batch_size_per_proc: int,
        gradient_accumulation_steps: int,
        world_size: int,
    ) -> None:
        """Log actual batch size per optimization step.

        Args:
        ----
            batch_size_per_proc (int): batch size per process.
            gradient_accumulation_steps (int): gradient accumulation steps.
            world_size (int): world size.
        """
        real_batch_size = batch_size_per_proc * gradient_accumulation_steps * world_size
        if real_batch_size == batch_size_per_proc:
            return

        self.log(
            (
                "Batch size:\n"
                f"--------------------------------------------------\n"
                f"            Batch size per process : {batch_size_per_proc}\n"
                f"       Gradient accumulation steps : {gradient_accumulation_steps}\n"
                f"               Number of processes : {world_size}\n"
                f"  ----------------------------------------------\n"
                f"  Batch size per optimization step : {real_batch_size}\n"
                f"--------------------------------------------------"
            )
        )

    def log_stuff(self, *to_log) -> None:
        """Log information in one function."""
        self.log_env()
        self.log_command_line()
        for obj in to_log:
            if isinstance(obj, DataLoader):
                self.log_dataset(obj)
            elif isinstance(obj, torch.nn.Module):
                self.log_model(obj)
            elif isinstance(obj, Optimizer):
                self.log_optimizer(obj)
            elif isinstance(obj, Namespace):
                self.log_args(obj)
            elif isinstance(obj, DictConfig):
                self.log_omegaconf(obj)

    """information accumulation funcs"""

    def update(self, **kwargs) -> None:
        """Update status."""
        self._collector.report_by_dict(kwargs)
        self.batches_done += 1

        self._steptimes.append(time.time() - self._step_start)

        # log
        if self._log_file is not None and (
            (self.batches_done == 1)
            or (self.batches_done % self._log_interval == 0)
            or (
                self.batches_done <= self._log_frequently_until
                and self.batches_done % 5 == 0
            )
            or (self.batches_done == self.max_iters)
        ):
            self._log_progress()

        if self.batches_done == self._log_nvidia_smi_at:
            # print gpu after some batches
            # for checking memory usage
            self.log_nvidia_smi()

        if self._profiler is not None:
            self._profiler.step()

        self._step_start = time.time()

    def _log_progress(self):
        """Log progress."""
        delta = self._collector.update()
        delta_str = []

        for key, value in delta.items():
            delta_str.append(self._delta_format.format(key=key, value=value))

        message_parts = [
            f"STEP: {self.batches_done} / {self.max_iters}",
            f'INFO: {", ".join(delta_str)}',
        ]

        # Memory usage
        if torch.cuda.is_available():
            _, global_total = torch.cuda.mem_get_info()
            local_usage = torch.cuda.memory_reserved() / global_total * 100
            message_parts.append(f"VRAM_used(%): {local_usage:.1f}")

        # ETA
        # NOTE: this ETA is not exact.
        #       dealed by avging multiple steps. (see rolling eta)
        duration = self._steptimes[-1]
        eta_sec = int((self.max_iters - self.batches_done) * duration)
        eta = datetime.timedelta(seconds=eta_sec)
        message_parts.append(f"ETA(sec): {eta}")

        # rolling eta for more stable ETA
        if len(self._steptimes) == self._steptimes.maxlen:
            rolling_duration = mean(self._steptimes)
            rolling_eta_sec = int(
                (self.max_iters - self.batches_done) * rolling_duration
            )
            rolling_eta = datetime.timedelta(seconds=rolling_eta_sec)
            message_parts.append(f"rolling_ETA(sec): {rolling_eta}")

        self.log(" ".join(message_parts))
        self.tb_add_scalars(**delta)

    def tb_add_scalars(self, **kwargs) -> None:
        """Add scalars to tensorboard."""
        for key, value in kwargs.items():
            self._tbwriter.add_scalar(key, value, self.batches_done)

    def tb_add_images(
        self,
        tag: str,
        image_tensor: torch.Tensor,
        normalize=True,
        value_range=(-1, 1),
        nrow=8,
        **mkgridkwargs,
    ) -> None:
        """Add image to tensorboard.

        Args:
        ----
            tag (str): tag.
            image_tensor (torch.Tensor): tensor of images.
            normalize (bool, optional): argument for make_grid(). Default: True.
            value_range (tuple, optional): argument for make_grid(). Default: (-1, 1).
            nrow (int, optional): argument for make_grid(). Default: 8.
            **mkgridkwargs: other keyword arguments for make_grid().
        """
        images = make_grid(
            image_tensor,
            normalize=normalize,
            value_range=value_range,
            nrow=nrow,
            **mkgridkwargs,
        )
        self._tbwriter.add_images(tag, images, self.batches_done)

    def dry_update(self, **kwargs):
        """Update accumulation values without updating iteration counts."""
        self._collector.report_by_dict(kwargs)
        self.tb_add_scalars(**kwargs)

    @contextmanager
    def stop_timer(self, verbose=False) -> None:
        """Context manager to stop the timer.

        Args:
        ----
            verbose (bool, optional): Boolean to enable logging. Default: False.

        Examples::
            >>> while not status.is_end():
            >>>     train_epoch(...)
            >>>     # this will stop the timer until exiting the with statement.
            >>>     with status.stop_timer():
            >>>         validate(...)
        """
        stop_start = time.time()
        yield
        duration = time.time() - stop_start
        if verbose:
            self.log(f"TIMER[STOPPED]: duration: {duration}")
        self._step_start += duration

    def is_end(self) -> None:
        """Have reached last batch."""
        return self.batches_done >= self.max_iters

    def _shutdown_logger(self) -> None:
        """Safely shutdown the loggers. This function will be automatically be called using atexit."""
        self.log("LOGGER: shutting down logger...")
        handlers = self._logger.handlers
        for handler in handlers:
            self._logger.removeHandler(handler)
            handler.close()

    def load_state_dict(self, state_dict: dict) -> None:
        """Fast forward training status by the given state_dict.

        Args:
        ----
            state_dict (dict): a dictionary made by status.state_dict().
        """
        # load
        self._collector = state_dict["collector"]
        self.batches_done = state_dict["batches_done"]
        self._steptimes = state_dict["steptimes"]

    def state_dict(self) -> dict:
        """Make a dictionary to save current training status.

        Returns
        -------
            dict: Dict containing the states.
        """
        return dict(
            collector=self._collector,
            batches_done=self.batches_done,
            steptimes=self._steptimes,
        )
