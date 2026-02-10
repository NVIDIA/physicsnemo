import inspect
import types
from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
import torch


def check_type(
    value: Any, expected_type: type | types.GenericAlias, name: str = ""
) -> None:
    """Perform runtime type checking for both simple and generic types.

    This function implements a sophisticated runtime type checker that validates
    values against Python type hints, including support for generic types like
    list[int], dict[str, float], and tuple[int, str]. It recursively validates
    nested structures and provides detailed error messages with the path to any
    type mismatches.

    ### Type Checking Behavior

    The function handles three categories of types:

    1. **Simple types** (int, str, MyClass, etc.): Uses isinstance() for validation
    2. **Generic types** (list[int], dict[str, int], etc.): Recursively validates
       both the container type and the types of its elements
    3. **Ellipsis type** (type(...)): Used in variable-length tuples like
       tuple[int, ...]; these are treated as "any number of elements allowed"

    ### Sequence Handling

    For sequences (list, tuple, etc.), the function distinguishes between:
    - **Fixed-length tuples**: When the number of type arguments equals the
      sequence length, each element is checked against its corresponding type.
      Example: tuple[int, str, float] expects exactly 3 elements with those types.
    - **Uniform sequences**: When there's a single type argument (optionally
      followed by ellipsis), all elements must match that type.
      Example: list[int] or tuple[str, ...] expect all elements to be of one type.

    Args:
        value: The value to type-check. Can be any Python object.
        expected_type: The expected type, either a simple type (like int) or a
            generic alias (like dict[str, float]). Supports nested generic types.
        name: Optional name for the value being checked, used in error messages
            to provide context. For nested structures, this builds up a path
            like "config['users'][0]['age']".

    Raises:
        TypeError: When the value doesn't match the expected type. The error
            message includes the path to the mismatched value if name is provided.
        ValueError: When the type checking logic encounters an unexpected
            condition, such as an invalid number of type arguments.
        NotImplementedError: When attempting to check a generic type that isn't
            yet supported (currently only Sequence and dict are implemented).

    Examples:
        >>> # Simple type checking
        >>> check_type(42, int)  # Passes
        >>> check_type("hello", int, "username")  # Raises TypeError

        >>> # Generic list checking
        >>> check_type([1, 2, 3], list[int])  # Passes
        >>> check_type([1, "two", 3], list[int])  # Raises TypeError

        >>> # Fixed-length tuple checking
        >>> check_type((1, "hello", 3.14), tuple[int, str, float])  # Passes
        >>> check_type((1, 2, 3), tuple[int, str, float])  # Raises TypeError

        >>> # Variable-length tuple checking
        >>> check_type((1, 2, 3, 4), tuple[int, ...])  # Passes

        >>> # Nested dictionary checking
        >>> data = {"users": [{"name": "Alice", "age": 30}]}
        >>> check_type(data, dict[str, list[dict[str, str | int]]])  # Passes

        >>> # Error message with path
        >>> check_type({"x": [1, "two"]}, dict[str, list[int]], "config")
        # Raises: TypeError: Expected 'config['x'][1]' to be a <class 'int'>, got a <class 'str'>.
    """
    ### Handle ellipsis type (used in variable-length tuples)
    if expected_type is type(...):
        return
    elif isinstance(expected_type, type):
        ### Simple type checking (int, str, custom classes, etc.)
        # This handles all non-generic types using Python's isinstance
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Expected {name!r} to be a {expected_type!r}, got a {type(value)!r}."
            )
    elif isinstance(expected_type, types.GenericAlias):
        ### Generic type checking (list[int], dict[str, int], tuple[int, str], etc.)
        # Extract the base type (e.g., list) and its type arguments (e.g., [int])
        origin_type: type = (  # ty: ignore[invalid-assignment]
            expected_type.__origin__
        )  # Base container type
        args_types: tuple[type | types.GenericAlias, ...] = (
            expected_type.__args__
        )  # Type parameters

        # First, recursively check that the value matches the container type
        check_type(value, origin_type, name=name)
        if issubclass(origin_type, Sequence):
            ### Sequence validation (list, tuple, etc.)
            for i, element in enumerate(value):
                # Determine the expected type for this element
                if len(args_types) == len(value):
                    ### Fixed-length tuple case: tuple[int, str, float]
                    # Each position has a specific type
                    expected_element_type = args_types[i]
                elif (len(args_types) == 1) or (
                    len(args_types) == 2 and args_types[1] is type(...)
                ):
                    ### Uniform sequence case: list[int] or tuple[int, ...]
                    # All elements have the same type
                    # The ellipsis in tuple[int, ...] means "any number of ints"
                    expected_element_type = args_types[0]
                else:
                    # Invalid type specification - neither fixed-length nor uniform
                    raise ValueError(
                        f"Expected {len(args_types)=} for {origin_type.__name__}, got {len(args_types)=}."
                    )

                # Recursively check this element, building up the path for error messages
                check_type(
                    value=element,
                    expected_type=expected_element_type,
                    name=f"{name}[{i}]",  # e.g., "config[0]" or "data['users'][2]"
                )
        elif issubclass(origin_type, dict):
            ### Dictionary validation
            # Dictionaries always have exactly 2 type arguments: [KeyType, ValueType]
            expected_key_type, expected_value_type = args_types

            # Check each key-value pair
            for k, v in value.items():
                # Validate the key type
                check_type(k, expected_key_type, f"{name} key {k!r}")
                # Validate the value type, with path showing the key
                check_type(v, expected_value_type, f"{name}[{k!r}]")
        else:
            ### Unsupported generic types
            # Currently only Sequence and dict are implemented
            # Could extend to support set[T], frozenset[T], etc.
            raise NotImplementedError(
                f"Type-checking not yet supported for {expected_type.__origin__.__name__} types."
            )
    else:
        ### Invalid expected_type parameter; this should not happen in normal usage
        raise ValueError(
            f"Got {type(expected_type)=!r}, expected a type or generic alias."
        )


def check_leaf_tensors(
    value: Any,
    func: Callable[[torch.Tensor], Any],
    name: str = "",
    func_name: str | None = None,
) -> dict[Any, list[str]]:
    """Recursively check that a function applied to all tensors returns the same value.

    This allows checking any property of tensors in a nested structure. It
    traverses arbitrarily nested data structures to find all PyTorch tensors,
    applies a user-defined function to each, and verifies that all results are
    identical.

    ### Traversal Behavior

    The function performs a depth-first traversal of the data structure:
    - **Tensors**: Treated as leaf nodes; func is applied and result is recorded
    - **Dictionaries**: Both keys and values are recursively checked
    - **Sequences**: All elements are recursively checked (excluding strings, bytes, or np.ndarrays)
    - **Other types**: Treated as leaves with no tensors

    ### Validation

    If the function returns different values for different tensors, a detailed error
    is raised showing the distribution of values and which tensors produced each value.
    This helps quickly identify inconsistencies in tensor properties.

    Args:
        value: The data structure to check. Can be any Python object, including
            nested combinations of dicts, lists, tuples containing torch tensors.
        func: A callable that takes a torch.Tensor and returns a hashable value.
            Common examples include lambda tensor: tensor.device, tensor.dtype,
            tensor.shape, tensor.requires_grad, etc.
        name: Optional name for the root structure, used to build descriptive
            paths in error messages. For nested structures, paths are built like
            "data['model'][0]['weights']" to precisely locate tensors.
        func_name: Optional human-readable name for the function being applied,
            used in error messages. If not provided, tries to use func.__name__
            or falls back to repr(func).

    Returns:
        A dictionary mapping function outputs to lists of tensor paths. If all
        tensors produce the same value, this dict will have a single key. Empty
        dict if no tensors are found. The paths help identify tensor locations.

    Raises:
        ValueError: When the function returns different values for different tensors.
            The error message includes a breakdown showing the distribution of values
            and which tensors produced each value.

    Examples:
        >>> import torch
        >>> # Check all tensors have same device
        >>> tensor1 = torch.randn(3, 4)
        >>> tensor2 = torch.randn(2, 5)
        >>> data = {"a": tensor1, "b": tensor2}
        >>> check_leaf_tensors(data, lambda t: t.device, "model")
        {device(type='cpu'): ['model[a]', 'model[b]']}

        >>> # Check all tensors have same shape - will fail
        >>> try:
        ...     check_leaf_tensors(data, lambda t: t.shape, "model", "shape")
        ... except ValueError as e:
        ...     print("Error:", e)
        Error: Expected all leaf tensors of 'model' to have the same shape.
        Found multiple different values:
            2 with torch.Size([3, 4]): model[a]
            1 with torch.Size([2, 5]): model[b]

        >>> # Check all tensors have same dtype
        >>> data = {
        ...     "floats": [torch.randn(2, 3), torch.randn(3, 4)],
        ...     "also_float": torch.zeros(5, 6)
        ... }
        >>> check_leaf_tensors(data, lambda t: t.dtype, "tensors", "dtype")
        {torch.float32: ['tensors[floats][0]', 'tensors[floats][1]', 'tensors[also_float]']}

        >>> # Check gradient requirements
        >>> x = torch.randn(2, 3, requires_grad=True)
        >>> y = torch.randn(3, 4, requires_grad=False)
        >>> check_leaf_tensors({"x": x, "y": y}, lambda t: t.requires_grad)
        # Raises ValueError showing gradient requirement mismatch
    """
    ### Determine the function description for error messages
    if func_name is None:
        func_name = getattr(func, "__name__", repr(func))
        if func_name == "<lambda>":
            try:
                func_name = inspect.getsource(func).strip()
                lambda_index = func_name.find("lambda")
                if lambda_index != -1:
                    func_name = func_name[lambda_index:]
            except (OSError, TypeError):
                pass

    ### Base case: tensor leaf
    if isinstance(value, torch.Tensor):
        result = func(value)
        return {result: [name]}

    ### Recursive case: composite structures
    results_by_value: dict[Any, list[str]] = defaultdict(list)

    if isinstance(value, dict):
        ### Dictionary: recurse into both keys and values
        for k, v in value.items():
            # Check if the key itself contains tensors (unusual but possible)
            for result, paths in check_leaf_tensors(
                value=k, func=func, name=f"{name} key {k!r}", func_name=func_name
            ).items():
                results_by_value[result].extend(paths)
            # Check the value
            for result, paths in check_leaf_tensors(
                value=v, func=func, name=f"{name}[{k!r}]", func_name=func_name
            ).items():
                results_by_value[result].extend(paths)

    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, np.ndarray)
    ):
        ### Sequence (list, tuple, etc.) but not string/bytes/array: recurse into elements
        for i, element in enumerate(value):
            for result, paths in check_leaf_tensors(
                value=element, func=func, name=f"{name}[{i}]", func_name=func_name
            ).items():
                results_by_value[result].extend(paths)

    ### Check for consistency at this level
    if len(results_by_value) > 1:
        # Format result distribution for error message
        result_summary = "\n".join(
            f"\t{len(paths)} with {result!r}: {', '.join(paths)}"
            for result, paths in results_by_value.items()
        )
        if name == "":
            name = "input"
        raise ValueError(
            f"Expected all leaf tensors of {name} to have the same {func_name}.\n"
            f"Found multiple different values:\n"
            f"{result_summary}"
        )

    return results_by_value
