import inspect
import re
from functools import cache
from pathlib import Path
from typing import Any, Literal

import git
import mlflow
import numpy as np
import torch
import yaml
from tensordict import TensorDict

import physicsnemo
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.mesh import Mesh


def disable_autotune_printing() -> None:
    """
    Disables the (extremely verbose) command-line output of `torch.compile(..., mode="max-autotune")`.
    """
    from torch._inductor import config, select_algorithm

    config.max_autotune_report_choices_stats = False
    select_algorithm.PRINT_AUTOTUNE = False


def sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name so it contains only allowed characters for MLflow:

    This replaces any character not in the set [A-Za-z0-9_.- :/] with a space.
    Leading or trailing whitespace is removed after replacement.

    Args:
        name: Original metric name that may contain special characters.

    Returns:
        Sanitized metric name with disallowed characters replaced by space, and whitespace trimmed.

    Examples:
        >>> sanitize_metric_name("ln(1+nut/nu)")
        'ln_1_nut_nu'
        >>> sanitize_metric_name("ΔU/|U_inf|")
        'U_U_inf'
        >>> sanitize_metric_name("C_F,shear")
        'C_F_shear'
    """
    import string

    allowed_chars = set(string.ascii_letters + string.digits + "_-. :")
    sanitized = "".join(c if c in allowed_chars else " " for c in name)
    while "  " in sanitized:
        sanitized = sanitized.replace("  ", " ")
    return sanitized.strip().replace(" ", "_")


@cache
def get_physicsnemo_pkg_info() -> dict[str, str | None]:
    """Get the git hash and version of the physicsnemo package.

    Returns:
        Dictionary with keys:
            - "version": Package version string (None if not available)
            - "git_hash": Git commit hash string (None if not in a git repository)
    """
    try:
        git_hash = git.Repo(search_parent_directories=True).head.commit.hexsha
    except git.InvalidGitRepositoryError:
        git_hash = None

    return {
        "version": getattr(physicsnemo, "__version__", None),
        "git_hash": git_hash,
    }


def extract_epoch(path: Path) -> int | None:
    """Extract epoch number from a checkpoint filename.

    Expects filenames matching the pattern 'ClassName.<epoch>.pt' where
    <epoch> is an integer.

    Args:
        path: Path to checkpoint file.

    Returns:
        Epoch number as integer, or None if filename doesn't match expected pattern.

    Examples:
        >>> extract_epoch(Path("Model.100.pt"))
        100
        >>> extract_epoch(Path("dir/MyModel.42.pt"))
        42
        >>> extract_epoch(Path("invalid_name.pt"))
        None
    """
    match = re.match(r".*\.(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else None


def get_latest_checkpoint_path(
    output_dir: Path, only_use_best: bool = False
) -> Path | None:
    """Find the checkpoint with the highest epoch number in an output directory.

    Searches for checkpoint files (*.pt) in both the 'models' and 'models/best_model'
    subdirectories of the given output directory. Determines recency by extracting
    epoch numbers from filenames.

    Args:
        output_dir: Root directory containing model checkpoints.
        only_use_best: Whether to only consider the best model checkpoint.

    Returns:
        Path to the checkpoint with highest epoch number, or None if no checkpoints
        are found.
    """
    models_dir = output_dir / "models"
    best_model_dir = models_dir / "best_model"

    checkpoint_dirs = [best_model_dir]
    if not only_use_best:
        checkpoint_dirs.append(models_dir)

    checkpoint_paths: list[Path] = []
    for directory in checkpoint_dirs:
        if directory.is_dir():
            checkpoint_paths.extend(directory.glob("*.pt"))

    if not checkpoint_paths:
        return None

    def sort_key(p: Path) -> int:
        epoch = extract_epoch(p)
        return -1 if epoch is None else epoch

    return max(
        checkpoint_paths,
        key=sort_key,
    )


def log_hyperparameters(
    log_dir: Path, model: torch.nn.Module, other_hyperparameters: dict[str, Any]
) -> None:
    """Log model and training hyperparameters to a YAML file.

    Extracts model constructor parameters by introspecting the model's __init__
    signature and matching against instance attributes. Converts complex objects
    (tensors, devices, etc.) to serializable representations.

    Args:
        log_dir: Directory where the hyperparameters.yaml file will be saved.
            Directory will be created if it doesn't exist.
        model: PyTorch model whose hyperparameters should be logged.
        other_hyperparameters: Additional hyperparameters to log (e.g., training
            configuration, optimizer settings, etc.).

    Side Effects:
        Creates log_dir if it doesn't exist and writes hyperparameters.yaml file.
    """

    def to_serializable(obj: Any) -> Any:
        """Recursively convert *obj* into a structure that can be handled by ``yaml.safe_dump``.

        This strips out complex Torch and distributed objects (``ProcessGroup``, devices, tensors, etc.)
        that cannot be pickled by the YAML representer. When an object cannot be converted in a loss-free
        manner, it is coerced to ``str(obj)`` so that at least a human-readable representation is kept.

        The implementation purposefully avoids importing heavy optional dependencies at the module level
        to keep import time negligible.
        """
        # Primitive types that YAML already knows how to handle
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # Handle common container types recursively
        if isinstance(obj, (list, tuple, set)):
            return [to_serializable(item) for item in obj]
        if isinstance(obj, dict):
            return {str(to_serializable(k)): to_serializable(v) for k, v in obj.items()}

        # pathlib.Path ➔ string
        if isinstance(obj, Path):
            return str(obj)

        try:
            if isinstance(obj, torch.Tensor):
                # For tensors, a nested list works well up to a reasonable size. For very large tensors, a
                # string representation is safer to avoid bloating the YAML file.
                return (
                    obj.tolist()
                    if obj.numel() <= 32
                    else f"Tensor(shape={tuple(obj.shape)})"
                )

            if isinstance(obj, torch.device):
                return str(obj)

            # torch.dtype has a nice string representation; anything else falls through
            if isinstance(obj, torch.dtype):
                return str(obj)

            if isinstance(obj, np.ndarray):
                return obj.tolist() if obj.size <= 32 else f"ndarray(shape={obj.shape})"

        except Exception as e:
            import warnings

            warnings.warn(f"Failed to serialize {obj} with error {e}")
            return str(obj)

        # Fallback – stringify the object
        return str(obj)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    constructor_params: list[str] = list(
        inspect.signature(type(model).__init__).parameters.keys()
    )[1:]  # Skip 'self'

    model_hyperparameters = {
        param: to_serializable(getattr(model, param))
        for param in constructor_params
        if hasattr(model, param)
    }
    other_hyperparameters = {
        k: to_serializable(v) for k, v in other_hyperparameters.items()
    }

    with open(log_dir / "hyperparameters.yaml", "w") as f:
        yaml.safe_dump(
            {
                "model": model_hyperparameters,
                **other_hyperparameters,
            },
            f,
            default_flow_style=False,
            indent=2,
            sort_keys=False,
        )

    mlflow.log_params(
        {
            **model_hyperparameters,
            **other_hyperparameters,
        }
    )


TransferrableType = (
    torch.Tensor
    | Mesh
    | TensorDict
    | list
    | tuple
    | dict
    | set
    | float
    | int
    | bool
    | complex
)


def to(
    data: TransferrableType, device: torch.device, dtype: torch.dtype | None = None
) -> TransferrableType:
    """Recursively transfer data structures to a PyTorch device.

    Supports nested data structures containing tensors and TensorDict-based
    objects (including Mesh). Preserves the structure of the input.

    Args:
        data: Data to transfer. Can be:
            - torch.Tensor or TensorDict (including Mesh, which is a tensorclass)
            - list/tuple/dict/set containing supported types (recursive)
        device: Target PyTorch device (e.g., torch.device('cuda:0')).
        dtype: Target dtype (e.g., torch.float32, torch.float64) or None (keep original dtype)

    Returns:
        Same structure as input with all tensors/transferrable objects moved
        to the specified device.

    Raises:
        NotImplementedError: If data contains unsupported types.

    Examples:
        >>> tensor = torch.randn(3, 4)
        >>> device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        >>> moved = to(tensor, device)
        >>> nested = {'a': [tensor, tensor], 'b': tensor}
        >>> moved_nested = to(nested, device)
    """
    if isinstance(data, (torch.Tensor, Mesh, TensorDict)):
        return data.to(device=device, dtype=dtype)
    elif isinstance(data, (float, int, bool, complex)):
        return torch.as_tensor(data, device=device)
    elif isinstance(data, list):
        return [to(item, device=device) for item in data]
    elif isinstance(data, tuple):
        return tuple(to(item, device=device) for item in data)
    elif isinstance(data, dict):
        return {k: to(v, device=device) for k, v in data.items()}
    elif isinstance(data, set):
        return {to(item, device=device) for item in data}
    else:
        raise NotImplementedError(
            f"`to_device` doesn't have a device-transfer recipe registered for {type(data)=!r}."
        )


def reduce_over_ranks(
    x: torch.Tensor,
    op: Literal["mean", "sum", "max", "min"] = "mean",
) -> torch.Tensor:
    """Reduce a tensor across all ranks using the specified operation.

    Args:
        x: Tensor to reduce. Modified in-place.
        op: Reduction operation. One of "mean", "sum", "max", "min".

    Returns:
        The reduced tensor (same object as x, modified in-place).
    """
    if not DistributedManager.is_initialized():
        raise RuntimeError(
            "Distributed manager should be initialized when using reduce_over_ranks"
        )
    dist = DistributedManager()

    if dist.world_size != 1:
        op_map = {
            "mean": torch.distributed.ReduceOp.AVG,
            "sum": torch.distributed.ReduceOp.SUM,
            "max": torch.distributed.ReduceOp.MAX,
            "min": torch.distributed.ReduceOp.MIN,
        }
        torch.distributed.barrier()
        torch.distributed.all_reduce(x, op=op_map[op])
        torch.distributed.barrier()

    return x


if __name__ == "__main__":
    print(get_physicsnemo_pkg_info())
