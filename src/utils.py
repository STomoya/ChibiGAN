"""Utilities
"""

from __future__ import annotations

import datetime
import glob
import importlib
import logging
import os
import re
import shutil
import sys
import types
from typing import Any, Callable, Tuple

from hydra import compose, initialize_config_dir
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

"""hydra
Handling configuration files without hydra.main()
"""


def get_hydra_config(
    config_dir: str, config_name: str, overrides: list[str] = sys.argv[1:]
):
    with initialize_config_dir(
        config_dir=to_absolute_path(config_dir), version_base=None
    ):
        cfg = compose(config_name, overrides=overrides)
    return cfg


def save_hydra_config(config: DictConfig, filename: str):
    with open(filename, "w") as fout:
        fout.write(OmegaConf.to_yaml(config))


"""easy dict"""


class EasyDict(dict):
    """dict that can access keys like attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


"""path"""


class Path(str):
    """pathlib.Path like class but is a string object and with additional methods"""

    @property
    def stem(self) -> str:
        return os.path.splitext(self.name)[0]

    @property
    def suffix(self) -> str:
        return os.path.splitext(self.name)[-1]

    @property
    def name(self) -> str:
        return os.path.basename(self) if self.isfile() else ""

    # "/" operator joins paths like pathlib.Path
    def __div__(self, other: str) -> "Path":
        if not isinstance(other, str):
            raise TypeError(
                f'unsupported operand type(s) for /: "{self.__class__.__name__}" and "{other.__class__.__name__}"'
            )
        return type(self)(os.path.join(self, other))

    # also enable when true div
    __truediv__ = __div__

    def mkdir(self) -> None:
        """make directory if self not exists"""
        if not self.exists():
            os.makedirs(self)

    def expanduser(self) -> "Path":
        return type(self)(os.path.expanduser(self))

    def glob(
        self,
        recursive: bool = False,
        filter_fn: Callable | None = None,
        sort: bool = False,
        sortkey: Callable | None = None,
    ) -> list["Path"]:
        glob_pattern = self / ("**/*" if recursive else "*")
        paths = glob.glob(glob_pattern, recursive=recursive)
        if isinstance(filter_fn, Callable):
            paths = [path for path in paths if filter_fn(path)]
        if sort:
            paths = sorted(paths, key=sortkey)
        paths = [type(self)(path) for path in paths]
        return paths

    def exists(self) -> bool:
        return os.path.exists(self)

    def isdir(self) -> bool:
        return os.path.isdir(self)

    def isfile(self) -> bool:
        return os.path.isfile(self)

    def resolve(self) -> "Path":
        return type(self)(os.path.realpath(os.path.abspath(self)))

    def dirname(self) -> "Path":
        return type(self)(os.path.dirname(self))

    # functions from shutil
    # Path('./somewhere').rmtree() == shutile.rmtree('./somewhere')
    copy = shutil.copy
    move = shutil.move
    rmtree = shutil.rmtree


class Folder(object):
    """Class for easily handling paths inside a root directory."""

    def __init__(
        self, root: str, identify: bool = False, identifier: str = None
    ) -> None:
        if identify:
            identifier = (
                identifier
                if identifier is not None
                else datetime.datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
            )
            root += "_" + identifier

        self._roots = EasyDict()
        self._roots.root = Path(root)

    def __getattr__(self, __name: str) -> Any:
        try:
            if __name in self._roots:
                return self._roots[__name]
            return self.__dict__[__name]
        except KeyError:
            raise AttributeError(__name)

    def add_children(self, **kwargs) -> None:
        for name, folder in kwargs.items():
            assert name != "_roots", '"_roots" is a reserved name. use another.'
            self._roots[name] = self._roots.root / folder

    def mkdir(self) -> None:
        for name in self._roots:
            self._roots[name].mkdir()

    def list(self) -> dict:
        return self._roots


"""others"""


"""
List of readable image extensions.
from https://github.com/pytorch/vision/blob/6512146e447b69cc9fb379eb05e447a17d7f6d1c/torchvision/datasets/folder.py#L242
"""
IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def is_image_file(path):
    """Returns true if the path is a PIL.Image.Image openable file"""
    ext = set([os.path.splitext(os.path.basename(path))[-1].lower()])
    return ext.issubset(IMG_EXTENSIONS)


def get_now_string(format: str = "%Y%m%d%H%M%S"):
    """Return current time as string"""
    return datetime.datetime.now().strftime(format)


def get_logger(
    name: str,
    filename: str = None,
    mode: str = "a",
    format: str = "%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s",
    auxiliary_handlers: list = None,
) -> logging.Logger:
    """setup and return logger

    Args:
        name (str): name of the logger. identical to logging.getLogger(name) if already called once with the same name.
        format (str, optional): logging format. Default: '%(asctime)s | %(name)s | %(filename)s | %(levelname)s | - %(message)s'.
        auxiliary_handlers (list, optional): Other user-defined handlers. Default: None

    Returns:
        logging.Logger: logger object.

    Examples:
        >>> logger = get_logger('logger-name')

        >>> # this should behave equivalent to logging.getLogger('logger-name')
        >>> # note that other args will be ignored in this situation.
        >>> get_logger('logger-name') == logger
        True
    """
    logger = logging.getLogger(name)

    if len(logger.handlers) > 0:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(format)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename, mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if auxiliary_handlers:
        for handler in auxiliary_handlers:
            logger.addHandler(handler)

    return logger


"""
Bellow is taken from: https://github.com/NVlabs/stylegan3/blob/583f2bdd139e014716fc279f23d362959bcc0f39/dnnlib/util.py#L233-L303
Creates an object using it's name and parameters.
Modified by: STomoya (https://github.com/STomoya)
"""


def _get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed).
    """

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [
        (".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)
    ]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            _get_obj_from_module(module, local_obj_name)  # may raise AttributeError
            return module, local_obj_name
        except Exception:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)  # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith(
                "No module named '" + module_name + "'"
            ):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            _get_obj_from_module(module, local_obj_name)  # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def _get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverses the object name and returns the last (rightmost) python object."""
    if obj_name == "":
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def _get_obj_by_name(name: str) -> Any:
    """Finds the python object with the given name."""
    module, obj_name = _get_module_from_obj_name(name)
    return _get_obj_from_module(module, obj_name)


def _call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Finds the python object with the given name and calls it as a function."""
    assert func_name is not None
    func_obj = _get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args: Any, class_name: str = None, **kwargs: Any) -> Any:
    """Finds the python class with the given name and constructs it with the given arguments.
    Arguments:
        *args: Any
            Positional arguments of the class
        class_name: str
            The name of the class. It should be the full name (like torch.optim.Adam).
        **kwargs: Any
            Keyword arguments of the class
    """
    return _call_func_by_name(*args, func_name=class_name, **kwargs)


"""
from: https://github.com/google/flax/blob/2387439a6f5c88627754905e6feadac4f33d9800/flax/training/checkpoints.py
naturally sorts the elements by numbers.
"""
UNSIGNED_FLOAT_RE = re.compile(r"[-+]?((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def natural_sort(iter):
    """sort files by numbers"""

    def maybe_num(s):
        return float(s) if UNSIGNED_FLOAT_RE.match(s) else s

    def split_keys(s):
        return [maybe_num(c) for c in UNSIGNED_FLOAT_RE.split(s)]

    return sorted(iter, key=split_keys)
