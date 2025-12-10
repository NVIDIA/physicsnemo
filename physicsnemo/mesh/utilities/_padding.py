import torch


def _pad_by_tiling_last(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Pads a tensor along its first dimension by tiling the last element.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to pad.
    size : int
        Target size for first dimension.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    return torch.cat(
        [tensor, torch.tile(tensor[-1:], (size - tensor.shape[0], 1))],
        dim=0,
    )


def _pad_with_value(tensor: torch.Tensor, size: int, value: float) -> torch.Tensor:
    """Pads a tensor along its first dimension with a constant value.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to pad.
    size : int
        Target size for first dimension.
    value : float
        Fill value for padding.

    Returns
    -------
    torch.Tensor
        Padded tensor.
    """
    return torch.cat(
        [
            tensor,
            torch.full(
                (size - tensor.shape[0], *tensor.shape[1:]),
                fill_value=value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=0,
    )
