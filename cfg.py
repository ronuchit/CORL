import copy
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore
from oc_extras import resolvers
from dm_control import suite
from omegaconf import DictConfig, ListConfig, OmegaConf

from contextlib import contextmanager
from typing import Any, Iterator, Union

import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig

CONFIG_PATH: Path = (Path(__file__).parent / "cfgs").resolve()

_is_hydra_initialized: bool = False


def get_domain_and_task(cfg_task: str) -> tuple[str, str]:
    """Split a "domain-task" string into a (domain, task) pair."""
    try:
        domain, task = cfg_task.split("-", 1)
    except ValueError:
        raise ValueError(f"Invalid task name: {cfg_task}")
    domain = dict(cup="ball_in_cup").get(domain, domain)
    assert (domain, task) in suite.ALL_TASKS, (domain, task)
    return domain, task


def get_load_from_step(load_from: Optional[str]) -> int:
    """Extract the number of steps from the path to a saved model.
    If this is the "final" model, return -1. If `load_from` is `None`,
    return -2.
    """
    if load_from is None:
        return -2
    model_name = Path(load_from).name
    if model_name == "final":
        return -1
    return int(model_name)


def get_task_specific_config(task_cfg: str) -> str:
    """Obtain the specific config file associated to the given task.
    :param task_cfg: String of the form "domain-task" (ex: cartpole-swingup)
    :returns: The domain if it has a corresponding config file under the "cfgs/tasks"
        folder, and "default" otherwise.
    """
    domain, _ = get_domain_and_task(task_cfg)
    task_cfg_path = CONFIG_PATH / "benchmark" / "dm_control" / "task" / f"{domain}.yaml"
    return domain if task_cfg_path.exists() else "default"


def expand_shape(shape: ListConfig, dim: int, x: int) -> ListConfig:
    assert isinstance(shape, ListConfig)
    shape = copy.copy(shape)
    shape[dim] += x
    return shape


def init_hydra() -> None:
    """Initialize Hydra / OmegaConf before we can load the config."""
    global _is_hydra_initialized

    if _is_hydra_initialized:
        return
    init_config_store()
    init_resolvers()
    _is_hydra_initialized = True


def init_config_store() -> None:
    cs = ConfigStore.instance()

    # In order to be able to use options like `modality=pixels` on the command-line, a
    # corresponding config group must exist. This is done here by creating a dummy node.
    for init_group in ["modality", "task"]:
        cs.store(
            group=init_group,
            name="__dummy__",
            node=OmegaConf.create({}),
        )


def init_resolvers() -> None:
    # Register resolvers from `oc_extras`.
    resolvers.register_new_resolvers()
    # Resolver that returns its first argument if not `None`, and its second otherwise.
    OmegaConf.register_new_resolver("default", lambda x, y: x if x is not None else y)
    # Used to evaluate numeric expressions, e.g. ${eval: 10_000 / ${action_repeat}}
    OmegaConf.register_new_resolver("eval", eval)
    # Used to fetch the task-specific config overriding some default settings.
    OmegaConf.register_new_resolver(
        "get_task_specific_config", get_task_specific_config
    )
    # Convenience resolver to generate a task title for logging purpose.
    OmegaConf.register_new_resolver(
        "get_task_title", lambda t: t.replace("-", " ").title()
    )
    # Convenience resolver to extract training steps for logging purpose.
    OmegaConf.register_new_resolver("get_load_from_step", get_load_from_step)
    # Check if a given value is set.
    OmegaConf.register_new_resolver("is_not_none", lambda x: x is not None)
    # Negation.
    OmegaConf.register_new_resolver("not", lambda x: not x)
    # Condition.
    OmegaConf.register_new_resolver(
        "if", lambda cond, val1, val2: val1 if cond else val2
    )
    # Map key to value from input dict.
    OmegaConf.register_new_resolver("map", lambda key, mapping: mapping[key])
    # Minimum over a list of values.
    OmegaConf.register_new_resolver("min", lambda *values: min(values))
    OmegaConf.register_new_resolver("expand_shape", expand_shape)


class ConfigHolder:
    """Wrapper around an OmegaConf config.

    This class is used to "hide" a config from
    `hydra.utils.instantiate()`, to prevent it from being automatically
    copied by Hydra (which prevents sharing the original config object).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    def get_config(self) -> DictConfig:
        return self._cfg


def assert_allclose_or_none(
    x: Optional[torch.Tensor], y: Optional[torch.Tensor]
) -> None:
    if x is None or y is None:
        assert x is None and y is None
    else:
        assert torch.allclose(x, y)


@contextmanager
def init_config(overrides: Union[list[str], dict[str, Any]]) -> Iterator[DictConfig]:
    """Context manager used to run code with a specific config.

    Config overrides may be passed either as a list of strings as would
    be used on the command-line, or as a (possibly nested) dictionary of
    option / value pairs.
    """
    if isinstance(overrides, dict):
        overrides = overrides_dict_to_list(overrides)
    overrides.append("cfg_hash=testing")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_PATH)):
        cfg = compose(
            config_name="default",
            overrides=overrides,
            return_hydra_config=True,
        )
        # The code below ensures that the Hydra config is available, which is required
        # for the `${hydra:}` resolver to work.
        # See https://github.com/facebookresearch/hydra/issues/2017
        hydra_cfg = HydraConfig.instance()
        prev = hydra_cfg.cfg  # backup previous config (typically: `None`)
        hydra_cfg.set_config(cfg)
        try:
            cfg.hydra = None  # delete the `hydra` key from our config
            yield cfg
        finally:
            hydra_cfg.cfg = prev  # restore previous config


def instantiate(cfg: DictConfig, *args: Any, **kwargs: Any) -> Any:
    """Wrap `hydra.utils.instantiate()` to provide additional features.

    This wrapper adds support for the `_node_cfg_` special argument.
    When present in the config, it should initially be set to `None`.
    This method then sets this keyword argument to be a `ConfigHolder`
    holding the configuration of the node being instantiated. This
    provides access to the original config object, that is usually lost
    when calling `hydra.utils.instantiate()`.
    """
    if "_node_cfg_" in cfg:
        assert cfg._node_cfg_ is None
        assert "_node_cfg_" not in kwargs
        kwargs["_node_cfg_"] = ConfigHolder(cfg)
    return hydra.utils.instantiate(cfg, *args, **kwargs)


def make_dir(dir_path: Path) -> Path:
    """Create directory if it does not already exist."""
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def overrides_dict_to_list(overrides: dict[str, Any]) -> list[str]:
    """Convert a dictionary {"a": x, "b": y} into a list ["a=x", "b=y"]"""
    list_overrides = []
    for k, v in overrides.items():
        if v is None:
            v = "null"
        elif isinstance(v, int):
            v = str(v)
        elif isinstance(v, str):
            # To be safe, we quote all strings => quotes must be escaped.
            if "'" in v:
                v = v.replace("'", r"\'")
            v = f"'{v}'"
        elif isinstance(v, dict):
            sub_overrides = overrides_dict_to_list(v)
            v = [f"{k}.{s}" for s in sub_overrides]
        else:
            raise NotImplementedError(
                f"`overrides_dict_to_list()` does not know how to convert option `{k}` "
                f"with value `{v}` of type {type(v)}"
            )

        if isinstance(v, list):
            list_overrides += v
        else:
            assert isinstance(v, str)
            list_overrides.append(f"{k}={v}")

    return list_overrides


def to_normalized_str(x: Any) -> str:
    """Utility function for consistent string representations.

    In particular this function is useful to ensure that integer and float values
    that represent the same number are represented with the same string (ex: both
    1.0 and 1 are represented as "1").

    All strings are lowercase.
    """
    if isinstance(x, str):
        return x.lower()
    elif isinstance(x, bool):
        return str(x).lower()
    elif isinstance(x, int):
        return str(x)
    elif isinstance(x, float):
        return str(int(x)) if x == int(x) else str(x)
    else:
        raise NotImplementedError(
            f"Input `{x}` to `to_normalized_str()` is of unsupported type: {type(x)}"
        )