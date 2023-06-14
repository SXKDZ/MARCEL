import itertools
import json
import logging
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn

from models.models_3d.utils.utils import read_value_json, update_json

ScaleDict = Union[Dict[str, float], Dict[str, torch.Tensor]]


class AutomaticFit:
    """
    All added variables are processed in the order of creation.
    """

    activeVar = None
    queue = None
    fitting_mode = False

    def __init__(self, variable, scale_file, name):
        self.variable = variable  # variable to find value for
        self.scale_file = scale_file
        self._name = name

        self._fitted = False
        self.load_maybe()

        # first instance created
        if AutomaticFit.fitting_mode and not self._fitted:

            # if first layer set to active
            if AutomaticFit.activeVar is None:
                AutomaticFit.activeVar = self
                AutomaticFit.queue = []  # initialize
            # else add to queue
            else:
                self._add2queue()

    def reset():
        AutomaticFit.activeVar = None
        AutomaticFit.all_processed = False

    def fitting_completed():
        return AutomaticFit.queue is None

    def set2fitmode():
        AutomaticFit.reset()
        AutomaticFit.fitting_mode = True

    def _add2queue(self):
        logging.debug(f"Add {self._name} to queue.")
        # check that same variable is not added twice
        for var in AutomaticFit.queue:
            if self._name == var._name:
                raise ValueError(
                    f"Variable with the same name ({self._name}) was already added to queue!"
                )
        AutomaticFit.queue += [self]

    def set_next_active(self):
        """
        Set the next variable in the queue that should be fitted.
        """
        queue = AutomaticFit.queue
        if len(queue) == 0:
            logging.debug("Processed all variables.")
            AutomaticFit.queue = None
            AutomaticFit.activeVar = None
            return
        AutomaticFit.activeVar = queue.pop(0)

    def load_maybe(self):
        """
        Load variable from file or set to initial value of the variable.
        """
        value = read_value_json(self.scale_file, self._name)
        if value is None:
            logging.info(
                f"Initialize variable {self._name}' to {self.variable.numpy():.3f}"
            )
        else:
            self._fitted = True
            logging.debug(f"Set scale factor {self._name} : {value}")
            with torch.no_grad():
                self.variable.copy_(torch.tensor(value))


class AutoScaleFit(AutomaticFit):
    """
    Class to automatically fit the scaling factors depending on the observed variances.

    Parameters
    ----------
        variable: tf.Variable
            Variable to fit.
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
    """

    def __init__(self, variable, scale_file, name):
        super().__init__(variable, scale_file, name)

        if not self._fitted:
            self._init_stats()

    def _init_stats(self):
        self.variance_in = 0
        self.variance_out = 0
        self.nSamples = 0

    def observe(self, x, y):
        """
        Observe variances for inut x and output y.
        The scaling factor alpha is calculated s.t. Var(alpha * y) ~ Var(x)
        """
        if self._fitted:
            return

        # only track stats for current variable
        if AutomaticFit.activeVar == self:
            nSamples = y.shape[0]
            self.variance_in += torch.mean(torch.var(x, dim=0)) * nSamples
            self.variance_out += torch.mean(torch.var(y, dim=0)) * nSamples
            self.nSamples += nSamples

    def fit(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if AutomaticFit.activeVar == self:
            if self.variance_in == 0:
                raise ValueError(
                    f"Did not track the variable {self._name}. Add observe calls to track the variance before and after."
                )

            # calculate variance preserving scaling factor
            self.variance_in = self.variance_in / self.nSamples
            self.variance_out = self.variance_out / self.nSamples

            ratio = self.variance_out / self.variance_in
            value = np.sqrt(1 / ratio, dtype="float32")
            logging.info(
                f"Variable: {self._name}, Var_in: {self.variance_in.numpy():.3f}, Var_out: {self.variance_out.numpy():.3f}, "
                + f"Ratio: {ratio:.3f} => Scaling factor: {value:.3f}"
            )

            # set variable to calculated value
            with torch.no_grad():
                self.variable.copy_(self.variable * value)
            update_json(self.scale_file, {self._name: float(self.variable.numpy())})
            self.set_next_active()  # set next variable in queue to active


class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.

    Parameters
    ----------
        scale_file: str
            Path to the json file where to store/load from the scaling factors.
        name: str
            Name of the scaling factor
    """

    def __init__(self, scale_file, name, device=None):
        super().__init__()

        self.scale_factor = torch.nn.Parameter(
            torch.tensor(1.0, device=device), requires_grad=False
        )
        self.autofit = AutoScaleFit(self.scale_factor, scale_file, name)

    def forward(self, x_ref, y):
        y = y * self.scale_factor
        self.autofit.observe(x_ref, y)

        return y


def _load_scale_dict(scale_file: Optional[Union[str, ScaleDict]]):
    """
    Loads scale factors from either:
    - a JSON file mapping scale factor names to scale values
    - a python dictionary pickled object (loaded using `torch.load`) mapping scale factor names to scale values
    - a dictionary mapping scale factor names to scale values
    """
    if not scale_file:
        return None

    if isinstance(scale_file, dict):
        if not scale_file:
            logging.warning("Empty scale dictionary provided to model.")
        return scale_file

    path = Path(scale_file)
    if not path.exists():
        raise ValueError(f"Scale file {path} does not exist.")

    scale_dict: Optional[ScaleDict] = None
    if path.suffix == ".pt":
        scale_dict = torch.load(path)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            scale_dict = json.load(f)

        if isinstance(scale_dict, dict):
            # old json scale factors have a comment field that has the model name
            scale_dict.pop("comment", None)
    else:
        raise ValueError(f"Unsupported scale file extension: {path.suffix}")

    if not scale_dict:
        return None

    return scale_dict


def load_scales_compat(
    module: nn.Module, scale_file: Optional[Union[str, ScaleDict]]
):
    scale_dict = _load_scale_dict(scale_file)
    if not scale_dict:
        return

    scale_factors = {
        module.name or name: (module, name)
        for name, module in module.named_modules()
        if isinstance(module, ScaleFactor)
    }
    logging.debug(
        f"Found the following scale factors: {[(k, name) for k, (_, name) in scale_factors.items()]}"
    )
    for name, scale in scale_dict.items():
        if name not in scale_factors:
            logging.warning(f"Scale factor {name} not found in model")
            continue

        scale_module, module_name = scale_factors[name]
        logging.debug(
            f"Loading scale factor {scale} for ({name} => {module_name})"
        )
        scale_module.set_(scale)


class _Stats(TypedDict):
    variance_in: float
    variance_out: float
    n_samples: int


IndexFn = Callable[[], None]


def _check_consistency(old: torch.Tensor, new: torch.Tensor, key: str):
    if not torch.allclose(old, new):
        raise ValueError(
            f"Scale factor parameter {key} is inconsistent with the loaded state dict.\n"
            f"Old: {old}\n"
            f"Actual: {new}"
        )


class ScaleFactor(nn.Module):
    scale_factor: torch.Tensor

    name: Optional[str] = None
    index_fn: Optional[IndexFn] = None
    stats: Optional[_Stats] = None

    def __init__(
        self,
        name: Optional[str] = None,
        enforce_consistency: bool = True,
    ):
        super().__init__()

        self.name = name
        self.index_fn = None
        self.stats = None

        self.scale_factor = nn.parameter.Parameter(
            torch.tensor(0.0), requires_grad=False
        )
        if enforce_consistency:
            self._register_load_state_dict_pre_hook(self._enforce_consistency)

    def _enforce_consistency(
        self,
        state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        if not self.fitted:
            return

        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key not in state_dict:
                continue

            input_param = state_dict[key]
            _check_consistency(old=param, new=input_param, key=key)

    @property
    def fitted(self):
        return bool((self.scale_factor != 0.0).item())

    @torch.jit.unused
    def reset_(self):
        self.scale_factor.zero_()

    @torch.jit.unused
    def set_(self, scale: Union[float, torch.Tensor]):
        if self.fitted:
            _check_consistency(
                old=self.scale_factor,
                new=torch.tensor(scale) if isinstance(scale, float) else scale,
                key="scale_factor",
            )
        self.scale_factor.fill_(scale)

    @torch.jit.unused
    def initialize_(self, *, index_fn: Optional[IndexFn] = None):
        self.index_fn = index_fn

    @contextmanager
    @torch.jit.unused
    def fit_context_(self):
        self.stats = _Stats(variance_in=0.0, variance_out=0.0, n_samples=0)
        yield
        del self.stats
        self.stats = None

    @torch.jit.unused
    def fit_(self):
        assert self.stats, "Stats not set"
        for k, v in self.stats.items():
            assert v > 0, f"{k} is {v}"

        self.stats["variance_in"] = (
            self.stats["variance_in"] / self.stats["n_samples"]
        )
        self.stats["variance_out"] = (
            self.stats["variance_out"] / self.stats["n_samples"]
        )

        ratio = self.stats["variance_out"] / self.stats["variance_in"]
        value = math.sqrt(1 / ratio)

        self.set_(value)

        stats = dict(**self.stats)
        return stats, ratio, value

    @torch.no_grad()
    @torch.jit.unused
    def _observe(self, x: torch.Tensor, ref: Optional[torch.Tensor] = None):
        if self.stats is None:
            logging.debug("Observer not initialized but self.observe() called")
            return

        n_samples = x.shape[0]
        self.stats["variance_out"] += (
            torch.mean(torch.var(x, dim=0)).item() * n_samples
        )

        if ref is None:
            self.stats["variance_in"] += n_samples
        else:
            self.stats["variance_in"] += (
                torch.mean(torch.var(ref, dim=0)).item() * n_samples
            )
        self.stats["n_samples"] += n_samples

    def forward(
        self,
        x: torch.Tensor,
        *,
        ref: Optional[torch.Tensor] = None,
    ):
        if self.index_fn is not None:
            self.index_fn()

        if self.fitted:
            x = x * self.scale_factor

        if not torch.jit.is_scripting():
            self._observe(x, ref=ref)

        return x