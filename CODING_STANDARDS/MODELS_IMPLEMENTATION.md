<!-- markdownlint-disable MD012 MD013 MD024 MD031 MD032 MD033 MD034 MD040 MD046 -->
<!-- MD012: Multiple consecutive blank lines -->
<!-- MD013: Line length -->
<!-- MD024: Multiple headings with the same content -->
<!-- MD031: Fenced code blocks should be surrounded by blank lines -->
<!-- MD032: Lists should be surrounded by blank lines -->
<!-- MD033: Inline HTML -->
<!-- MD034: Bare URL used -->
<!-- MD040: Fenced code blocks should have a language specified -->
<!-- MD046: Code block style -->

# MODELS_IMPLEMENTATION - Coding Standards

## Overview

This document defines the coding standards and best practices for implementing
model classes in the PhysicsNeMo repository. These rules are designed to ensure
consistency, maintainability, and high code quality across all model
implementations.

**Important:** These rules are enforced as strictly as possible. Deviations
from these standards should only be made when absolutely necessary and must be
documented with clear justification in code comments and approved during code
review.

## Document Organization

This document is structured in two main sections:

1. **Rule Index**: A quick-reference table listing all rules with their IDs,
   one-line summaries, and the context in which they apply. Use this section
   to quickly identify relevant rules when implementing or reviewing code.

2. **Detailed Rules**: Comprehensive descriptions of each rule, including:
   - Clear descriptions of what the rule requires
   - Rationale explaining why the rule exists
   - Examples demonstrating correct implementation
   - Anti-patterns showing common mistakes to avoid

## How to Use This Document

- **When creating new models**: Review all rules before starting implementation,
  paying special attention to rules MOD-000 through MOD-003.
- **When reviewing code**: Use the Rule Index to quickly verify compliance with
  all applicable rules.
- **When refactoring**: Ensure refactored code maintains or improves compliance
  with these standards.
- **For AI agents that generate code**: This document is formatted for easy parsing. Each rule has
  a unique ID and structured sections (Description, Rationale, Example,
  Anti-pattern) that can be extracted and used as context. When generating code
  based on a rule, an AI agent should explicitly quote the rule ID that it is
  following, and explicitly quote the relevant extract from the rule that it is
  using as context. For example, "Following rule MOD-000, the new model class
  should be ..."
- **For AI agents that review code**: When reviewing code, the AI agent should
  explicitly identify which rules are violated by the code, and provide a clear
  explanation of why the code violates the rule. The AI agent should explicitly
  quote the rule ID that the code is violating, and explicitly quote the relevant
  extract from the rule that it is using as context. For example, "Code violates
  rule MOD-000, because the new model class is not..."

## Rule Index

| Rule ID | Summary | Apply When |
|---------|---------|------------|
| [`MOD-000a`](#mod-000a-reusable-layersblocks-belong-in-physicsnemonn) | Reusable layers/blocks belong in physicsnemo.nn (stored in physicsnemo/nn/module) | Creating or refactoring reusable layer classes |
| [`MOD-000b`](#mod-000b-complete-models-belong-in-physicsnemomodels) | Complete models belong in physicsnemo.models | Creating or refactoring complete model classes |
| [`MOD-001`](#mod-001-use-physicsnemomodule-as-model-base-classes) | Use physicsnemo.Module as model base classes | Creating or refactoring new model classes |
| [`MOD-002a`](#mod-002a-new-models-and-layers-belong-in-physicsnemoexperimental) | New models and layers belong in physicsnemo.experimental | Creating new model or layer classes |
| [`MOD-002b`](#mod-002b-add-deprecation-warnings-to-deprecating-model-class) | Add deprecation warnings to deprecating model class | Deprecating existing model classes |
| [`MOD-002c`](#mod-002c-remove-deprecated-model-from-codebase) | Remove deprecated model from codebase | Removing deprecated models after warning period |
| [`MOD-003a`](#mod-003a-missing-or-incomplete-docstring-for-modellayer-code) | Missing or incomplete docstring for model/layer code | Creating or editing any model or layer code |
| [`MOD-003b`](#mod-003b-docstring-must-use-raw-string-prefix-r) | Docstring must use raw string prefix r""" | Writing any model or method docstring |
| [`MOD-003c`](#mod-003c-missing-required-class-docstring-sections) | Missing required class docstring sections | Writing class docstrings |
| [`MOD-003d`](#mod-003d-missing-required-method-docstring-sections) | Missing required method docstring sections | Writing method docstrings |
| [`MOD-003e`](#mod-003e-tensor-shapes-must-use-latex-math-notation) | Tensor shapes must use LaTeX math notation | Documenting tensors in docstrings |
| [`MOD-003f`](#mod-003f-callback-functions-must-have-code-block-specification) | Callback functions must have code-block specification | Documenting callback function parameters |
| [`MOD-003g`](#mod-003g-inline-code-must-use-double-backticks) | Inline code must use double backticks | Writing inline code in docstrings |
| [`MOD-003h`](#mod-003h-parameters-must-be-documented-on-single-line) | Parameters must be documented on single line | Documenting function/method parameters |
| [`MOD-003i`](#mod-003i-docstrings-should-include-cross-references) | Docstrings should include cross-references | Writing comprehensive docstrings |
| [`MOD-003j`](#mod-003j-docstrings-should-include-examples-section) | Docstrings should include Examples section | Writing model class docstrings |
| [`MOD-003k`](#mod-003k-add-high-level-comments-for-complex-tensor-operations) | Add high-level comments for complex tensor operations | Writing model code with complex tensor operations |
| [`MOD-004`](#mod-004-model-code-is-not-self-contained) | Model code is not self-contained | Organizing or refactoring model code |
| [`MOD-005`](#mod-005-invalid-or-missing-tensor-shape-validation-logic) | Invalid or missing tensor shape validation logic | Implementing model forward or public methods |
| [`MOD-006`](#mod-006-invalid-or-missing-jaxtyping-tensor-annotations-in-public-function-signature) | Invalid or missing jaxtyping tensor annotations in public function signature | Adding type hints to model methods |
| [`MOD-007a`](#mod-007a-cannot-add-required-parameters-without-defaults) | Cannot add required parameters without defaults | Modifying production model signatures |
| [`MOD-007b`](#mod-007b-cannot-remove-or-rename-parameters-without-compat-mapper) | Cannot remove or rename parameters without compat mapper | Modifying production model signatures |
| [`MOD-007c`](#mod-007c-cannot-change-return-types-of-public-methods) | Cannot change return types of public methods | Modifying production model method signatures |
| [`MOD-008a`](#mod-008a-model-missing-constructorattributes-tests) | Model missing constructor/attributes tests | Adding CI tests for models |
| [`MOD-008b`](#mod-008b-model-missing-non-regression-test-with-reference-data) | Model missing non-regression test with reference data | Adding CI tests for models |
| [`MOD-008c`](#mod-008c-model-missing-checkpoint-loading-test) | Model missing checkpoint loading test | Adding CI tests for models |
| [`MOD-009`](#mod-009-avoid-string-based-class-selection-in-model-constructors) | Avoid string-based class selection in model constructors | Designing model constructor APIs |
| [`MOD-010`](#mod-010-avoid-splatted-kwargs-in-model-constructors) | Avoid splatted kwargs in model constructors | Designing model constructor APIs |
| [`MOD-011`](#mod-011-use-proper-optional-dependency-handling) | Use proper optional dependency handling | Implementing models with optional dependencies |

---

## Detailed Rules

### MOD-000a: Reusable layers/blocks belong in physicsnemo.nn

**Description:**

Reusable layers that are the building blocks of more complex architectures
should live in `physicsnemo/nn/module` and be re-exported from
`physicsnemo/nn/__init__.py` so users can still import them from
`physicsnemo.nn`. Those include, for instance, `FullyConnected`, various variants
of attention layers, and `UNetBlock` (a block of a U-Net).

All layers that are directly exposed to the user should be imported in
`physicsnemo/nn/__init__.py`, such that they can be used as follows:

```python
from physicsnemo.nn import MyLayer
```

The only exception to this rule is for layers that are highly specific to a
single example. In this case, it may be acceptable to place them in a module
specific to the example code, such as `examples/<example_name>/utils/nn.py`.

**Rationale:**

Ensures consistency in the organization of reusable layers in the repository.
Keeping all reusable components in a single location makes them easy to find
and promotes code reuse across different models.

**Example:**

```python
# Good: Reusable layer in physicsnemo/nn/module/attention_layers.py
class MultiHeadAttention(Module):
    """A reusable attention layer that can be used in various architectures."""
    pass

# Good: Import in physicsnemo/nn/__init__.py
from physicsnemo.nn import MultiHeadAttention

# Good: Example-specific layer in examples/weather/utils/nn.py
class WeatherSpecificLayer(Module):
    """Layer highly specific to the weather forecasting example."""
    pass
```

**Anti-pattern:**

```python
# WRONG: Reusable layer placed in physicsnemo/models/
# File: physicsnemo/models/attention.py
class MultiHeadAttention(Module):
    """Should be in physicsnemo/nn/module/ not physicsnemo/models/"""
    pass
```

---

### MOD-000b: Complete models belong in physicsnemo.models

**Description:**

More complete models, composed of multiple layers and/or other sub-models,
should go into `physicsnemo/models`. All models that are directly exposed to
the user should be imported in `physicsnemo/models/__init__.py`, such that they
can be used as follows:

```python
from physicsnemo.models import MyModel
```

The only exception to this rule is for models that are highly specific to a
single example. In this case, it may be acceptable to place them in a module
specific to the example code, such as `examples/<example_name>/utils/nn.py`.

**Rationale:**

Ensures consistency and clarity in the organization of models in the repository,
in particular a clear separation between reusable layers and more complete
models that are applicable to a specific domain or specific data modality.

**Example:**

```python
# Good: Complete model in physicsnemo/models/transformer.py
class TransformerModel(Module):
    """A complete transformer model composed of attention and feedforward layers."""
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(...)
        self.ffn = FeedForward(...)

# Good: Import in physicsnemo/models/__init__.py
from physicsnemo.models.transformer import TransformerModel
```

**Anti-pattern:**

```python
# WRONG: Complete model placed in physicsnemo/nn/module/
# File: physicsnemo/nn/module/transformer_model.py
class TransformerModel(Module):
    """Should be in physicsnemo/models/ not physicsnemo/nn/module/"""
    pass
```

---

### MOD-001: Use physicsnemo.Module as model base classes

**Description:**

All model classes must inherit from `physicsnemo.Module`. Direct subclasses of
`torch.nn.Module` are not allowed. Direct subclasses of `physicsnemo.Module`
are allowed (note that `physicsnemo.Module` is a subclass of `torch.nn.Module`).
Ensure proper initialization of parent classes using `super().__init__()`. Pass
the `meta` argument to the `super().__init__()` call if appropriate, otherwise
set it manually with `self.meta = meta`.

**Rationale:**
Ensures invariants and functionality of the `physicsnemo.Module` class for all
models. In particular, instances of `physicsnemo.Module` benefit from features
that are not available in `torch.nn.Module` instances. Those include serialization
for checkpointing and loading modules and submodules, versioning system to
handle backward compatibility, as well as ability to be registered in the
`physicsnemo.registry` for easy instantiation and use in any codebase.

**Example:**

```python
from physicsnemo import Module

class MyModel(Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(meta=MyModelMetaData())
        self.linear = nn.Linear(input_dim, output_dim)
```

**Anti-pattern:**

```python
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        self.linear = nn.Linear(input_dim, output_dim)
```

---

### MOD-002a: New models and layers belong in physicsnemo.experimental

**Description:**

For the vast majority of models, new classes are created in
`physicsnemo/experimental/models` (including reusable layers). The
`experimental` folder is used to store models that are still under development
(beta or alpha releases), where backward compatibility is not guaranteed.

One exception is when the developer is highly confident that the model is
sufficiently mature and applicable to many domains or use cases. In this case
the model class can be created in `physicsnemo/nn/module` or
`physicsnemo/models` directly (and re-exported from `physicsnemo/nn`), and
backward compatibility is guaranteed.

Another exception is when the model class is highly specific to a single
example. In this case, it may be acceptable to place it in a module specific to
the example code, such as `examples/<example_name>/utils/nn.py`.

After staying in experimental for a sufficient amount of time (typically at
least 1 release cycle), the model class can be promoted to production. It is
then moved to `physicsnemo/nn/module` or `physicsnemo/models`, based on whether
it's a reusable layer (MOD-000a) or complete model (MOD-000b). During the
production stage, backward compatibility is guaranteed.

**Note:** Per MOD-008a, MOD-008b, and MOD-008c, it is forbidden to move a model
out of the experimental stage/directory without the required CI tests.

**Rationale:**

The experimental stage allows rapid iteration without backward compatibility
constraints, enabling developers to refine APIs based on user feedback. This
protects users from unstable APIs while allowing innovation.

**Example:**

```python
# Good: New experimental model
# File: physicsnemo/experimental/models/new_diffusion.py
class DiffusionModel(Module):
    """New diffusion model under active development. API may change."""
    pass

# Good: After 1+ release cycles, promoted to production
# File: physicsnemo/models/diffusion.py (moved from experimental/)
class DiffusionModel(Module):
    """Stable diffusion model with backward compatibility guarantees."""
    pass
```

**Anti-pattern:**

```python
# WRONG: New model directly in production folder
# File: physicsnemo/models/brand_new_model.py (should be in experimental/ first)
class BrandNewModel(Module):
    """Skipped experimental stage - risky for stability"""
    pass
```

---

### MOD-002b: Add deprecation warnings to deprecating model class

**Description:**

For a model class being deprecated in `physicsnemo/nn/module` or
`physicsnemo/models`, the developer must add warning messages indicating that
the model class is
deprecated and will be removed in a future release.

The warning message should be clear and concise, explaining why the model class
is being deprecated and what the user should do instead. The deprecation message
must be added to both:
1. The docstring using `.. deprecated::` directive
2. Runtime using `warnings.warn(..., DeprecationWarning)`

The developer is free to choose the mechanism to raise the deprecation warning.
A model class cannot be deprecated without staying in the pre-deprecation stage
for at least 1 release cycle before it can be deleted (see MOD-002c).

**Rationale:**

Ensures users have sufficient time to migrate to newer alternatives, preventing
breaking changes that could disrupt their workflows. This graduated approach
balances innovation with stability.

**Example:**

```python
# Good: Pre-deprecation with proper warnings
# File: physicsnemo/models/old_diffusion.py
class DiffusionModel(Module):
    """
    Legacy diffusion model.

    .. deprecated:: 0.5.0
        ``OldDiffusionModel`` is deprecated and will be removed in version 0.7.0.
        Use :class:`~physicsnemo.models.NewDiffusionModel` instead.
    """
    def __init__(self):
        import warnings
        warnings.warn(
            "OldDiffusionModel is deprecated. Use NewDiffusionModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()
```

**Anti-pattern:**

```python
# WRONG: No runtime warning
# File: physicsnemo/models/old_model.py
class OldModel(Module):
    """Will be removed next release."""  # Docstring mentions it but no runtime warning
    def __init__(self):
        # Missing: warnings.warn(..., DeprecationWarning)
        super().__init__()

# WRONG: Deprecation without sufficient warning period
# (Model deprecated and removed in same release)
```

---

### MOD-002c: Remove deprecated model from codebase

**Description:**

After staying in the pre-deprecation stage for at least 1 release cycle, the
model class is considered deprecated and can be deleted from the codebase.

A model class cannot be deleted without first spending at least 1 release cycle
in the pre-deprecation stage with proper deprecation warnings (see MOD-002b).

**Rationale:**

This ensures users have sufficient warning and time to migrate their code to
newer alternatives. Premature deletion of models would break user code without
adequate notice, violating the framework's commitment to stability.

**Example:**

```python
# Good: Proper deprecation timeline
# v0.5.0: Added deprecation warnings (Stage 3 - pre-deprecation)
# v0.6.0: Model can be safely removed (Stage 4 - deprecation)
# File: physicsnemo/models/old_diffusion.py - DELETED
```

**Anti-pattern:**

```python
# WRONG: Deleting model without deprecation period
# v0.5.0: Model exists without warnings
# v0.6.0: Model deleted - BREAKS USER CODE!

# WRONG: Breaking changes without deprecation
# File: physicsnemo/models/diffusion.py
class DiffusionModel(Module):
    def __init__(self, new_required_param):  # Breaking change!
        # Changed API without deprecation warning - breaks user code
        pass
```

---

### MOD-003a: Missing or incomplete docstring for model/layer code

**Description:**

Every new model or modification of any model code should be documented with a
comprehensive docstring following all the sub-rules MOD-003b through MOD-003k.
All docstrings should be written in the NumPy style and adopt formatting to be
compatible with our Sphinx restructured text (RST) documentation.

**Rationale:**

Comprehensive and well-formatted documentation is essential for scientific
software. It enables users to understand model capabilities, expected inputs,
and outputs without inspecting source code.

**Example:**

```python
class MyEncoder(Module):
    r"""
    A simple encoder network.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    output_dim : int
        Dimension of output features.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.

    Examples
    --------
    >>> model = MyEncoder(input_dim=784, output_dim=128)
    >>> x = torch.randn(32, 784)
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 128])
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: Missing all required sections
class BadEncoder(Module):
    '''A simple encoder.'''  # Wrong quotes, no sections
    pass
```

---

### MOD-003b: Docstring must use raw string prefix r"""

**Description:**

Each docstring should be prefixed with `r"""` (not `"""` or `'''`). The `r`
prefix creates a raw string that prevents Python from interpreting backslashes,
which is essential for LaTeX math notation to render correctly in Sphinx
documentation.

**Rationale:**

LaTeX commands in docstrings use backslashes (e.g., `\math`, `\text`). Without
the raw string prefix, Python interprets these as escape sequences, breaking the
documentation rendering.

**Example:**

```python
class MyModel(Module):
    r"""
    A model with LaTeX notation.

    Parameters
    ----------
    dim : int
        Dimension :math:`D` of input features.
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: Using ''' instead of r"""
class MyModel(Module):
    '''
    A model with LaTeX notation.
    '''
    pass
```

---

### MOD-003c: Missing required class docstring sections

**Description:**

The class docstring should at least contain three sections: `Parameters`,
`Forward`, and `Outputs`. The forward method should be documented in the
docstring of the model class, instead of being in the docstring of the forward
method itself. A docstring for the forward method is still possible but it
should be concise and to the point.

Other sections such as `Notes`, `Examples`, or `..important::` or
`..code-block::python`
are possible. Other sections are not recognized by our Sphinx restructured text
(RST) documentation and are prohibited.

**Rationale:**

Standardized sections ensure documentation is consistent and complete across all
models. The Forward and Outputs sections in the class docstring provide a
centralized place to document the model's primary behavior, making it easier for
users to understand the model's API.

**Example:**

```python
class MyModel(Module):
    r"""
    A simple encoder model.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, D_{in})`.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, D_{out})`.
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: Missing required sections
class BadModel(Module):
    r"""
    A simple encoder model.

    No proper sections defined.
    """
    pass
```

---

### MOD-003d: Missing required method docstring sections

**Description:**

All methods should be documented with a docstring, with at least a `Parameters`
section and a `Returns` section. Other sections such as `Notes`, `Examples`, or
`..important::` or `..code-block:: python` are possible. Other sections are not
recognized by our Sphinx documentation and are prohibited.

Note: The forward method is a special case - its full documentation should be in
the class docstring (see MOD-003c), though a concise forward method docstring is
permitted.

**Rationale:**

Complete method documentation ensures users understand how to call methods and
what to expect in return. Standardized sections make documentation consistent
and easier to parse for both humans and AI agents.

**Example:**

```python
def compute_loss(
    self,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute mean squared error loss.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values of shape :math:`(B, D)`.
    target : torch.Tensor
        Target values of shape :math:`(B, D)`.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    return torch.nn.functional.mse_loss(pred, target)
```

**Anti-pattern:**

```python
# WRONG: No docstring
def helper_method(self, x):
    return x * 2

# WRONG: Using wrong section names
def compute_loss(self, pred, target):
    """
    Args:
        pred: predictions
    Returns:
        loss
    """
    pass
```

---

### MOD-003e: Tensor shapes must use LaTeX math notation

**Description:**

All tensors should be documented with their shape, using LaTeX math notation
such as `:math:`(N, C, H_{in}, W_{in})``. There is flexibility for naming the
dimensions, but the math format should be enforced.

Our documentation is rendered using LaTeX, and supports a rich set of LaTeX
commands, so it is recommended to use LaTeX commands whenever possible for
mathematical variables in the docstrings. The mathematical notations should be
to some degree consistent with the actual variable names in the code.

**Rationale:**

LaTeX math notation ensures tensor shapes render correctly and consistently in
Sphinx documentation. This is critical for scientific software where precise
mathematical notation is expected. Plain text shapes don't render properly and
can be ambiguous.

**Example:**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    r"""
    Process input tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape :math:`(B, C, H_{in}, W_{in})` where :math:`B` is batch
        size, :math:`C` is channels, and :math:`H_{in}, W_{in}` are spatial dims.

    Returns
    -------
    torch.Tensor
        Output of shape :math:`(B, C_{out}, H_{out}, W_{out})`.
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: Not using :math: notation
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Input of shape (B, C, H, W)  # Missing :math:`...`
    """
    pass
```

---

### MOD-003f: Callback functions must have code-block specification

**Description:**

For arguments or variables that are callback functions (e.g. Callable), the
docstring should include a clear separated `..code-block::` that specifies the
required signature and return type of the callback function. This is not only
true for callback functions, but for any type of parameters or arguments that
has some complex type specification or API requirements.

The explanation code block should be placed in the top or bottom section of the
docstrings, but not in the `Parameters` or `Forward` or `Outputs` sections, for
readability and clarity.

**Rationale:**

Callback functions have complex type signatures that are difficult to express
clearly in the Parameters section alone. A dedicated code-block provides a clear
visual reference for the expected signature, making it much easier for users to
implement compatible callbacks.

**Example:**

```python
class MyModel(Module):
    r"""
    Model with callback function.

    .. code-block:: python

        def preprocess_fn(x: torch.Tensor) -> torch.Tensor:
            '''Preprocessing function signature.'''
            ...
            return y

    where ``x`` is input of shape :math:`(B, D_{in})` and ``y`` is output
    of shape :math:`(B, D_{out})`.

    Parameters
    ----------
    preprocess_fn : Callable[[torch.Tensor], torch.Tensor], optional
        Optional preprocessing function. See code block above for signature.
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: No code-block specification
class MyModel(Module):
    r"""
    Parameters
    ----------
    preprocess_fn : Callable, optional
        Preprocessing function.  # No specification!
    """
    pass
```

---

### MOD-003g: Inline code must use double backticks

**Description:**

Inline code should be formatted with double backticks, such as ``my_variable``.
Single backticks are not allowed as they don't render properly in our Sphinx
documentation.

**Rationale:**

Sphinx uses reStructuredText, which requires double backticks for inline code
literals. Single backticks are interpreted differently and don't produce the
expected code formatting in the rendered documentation.

**Example:**

```python
class MyModel(Module):
    r"""
    Model with inline code references.

    If ``True``, enables dropout. Set ``model.training`` to control behavior.
    The parameter ``hidden_dim`` controls layer size.

    Parameters
    ----------
    hidden_dim : int
        Size of hidden layer. Access via ``self.hidden_dim``.
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: Using single backticks
class MyModel(Module):
    r"""
    If `True`, enables dropout.  # WRONG
    """
    pass
```

---

### MOD-003h: Parameters must be documented on single line

**Description:**

All parameters should be documented with their type and default values on a
single line, following the NumPy docstring style format:

```
parameter_name : type, optional, default=value
```

The description then follows on the next line(s), indented.

**Rationale:**

This standardized format makes parameter documentation consistent and easy to
parse. It provides all key information (name, type, optionality, default) at a
glance, improving readability.

**Example:**

```python
class MyModel(Module):
    r"""
    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dim : int, optional, default=128
        Dimension of hidden layer.
    dropout : float, optional, default=0.1
        Dropout probability.
    """
    pass
```

**Anti-pattern:**

```python
# WRONG: Type and default not on same line
class MyModel(Module):
    r"""
    Parameters
    ----------
    hidden_dim : int
        optional, default=128  # Should be on line above
        Dimension of hidden layer.
    """
    pass
```

---

### MOD-003i: Docstrings should include cross-references

**Description:**

When possible, docstrings should use links to other docstrings using Sphinx
cross-reference syntax:
- Classes: `:class:`~physicsnemo.models.some_model.SomeModel``
- Functions: `:func:`~physicsnemo.utils.common_function``
- Methods: `:meth:`~physicsnemo.models.some_model.SomeModel.some_method``

When referencing external resources, such as papers, websites, or other
documentation, docstrings should use links to the external resource in the
format `some link text <some_url>`_.

**Rationale:**

Cross-references create a navigable documentation structure where users can
easily jump between related classes, methods, and functions. External links
provide context and attribution for algorithms and techniques.

**Example:**

```python
class MyEncoder(Module):
    r"""
    Encoder using attention.

    Based on `Transformer Architecture <https://arxiv.org/abs/1706.03762>`_.
    See :class:`~physicsnemo.nn.MultiHeadAttention` for attention details.

    Parameters
    ----------
    activation : str
        Activation function. See :func:`~torch.nn.functional.relu`.
    """
    pass
```

**Anti-pattern:**

```python
# Not wrong, but missing opportunities for useful links
class MyEncoder(Module):
    r"""
    Uses MultiHeadAttention.  # Could link to class
    Based on Transformer paper.  # Could link to paper
    """
    pass
```

---

### MOD-003j: Docstrings should include Examples section

**Description:**

Docstrings are strongly encouraged to have an `Examples` section that
demonstrates basic construction and usage of the model. These example sections
serve as both documentation and tests, as our CI system automatically tests
these code sections for correctness when present.

Examples should be executable Python code showing typical use cases, including
model instantiation, input preparation, and forward pass execution. The examples
should use realistic tensor shapes and demonstrate key features of the model.

**Rationale:**

Example sections provide immediate value to users by showing concrete usage
patterns. By automatically testing these examples in CI, we ensure that
documentation stays synchronized with code and that examples remain correct as
the codebase evolves.

**Example:**

```python
class MyEncoder(Module):
    r"""
    A simple encoder network.

    Parameters
    ----------
    input_dim : int
        Dimension of input features.

    Examples
    --------
    >>> import torch
    >>> from physicsnemo.models import MyEncoder
    >>> model = MyEncoder(input_dim=784, output_dim=128)
    >>> x = torch.randn(32, 784)
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 128])
    """
    pass
```

**Anti-pattern:**

```python
# Not wrong, but discouraged - no Examples section
class MyEncoder(Module):
    r"""
    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    """
    pass
```

---

### MOD-003k: Add high-level comments for complex tensor operations

**Description:**

Model code that involves complex tensor operations should include high-level
comments that explain what blocks of code accomplish semantically. One-line
comments every few lines of tensor operations is sufficient.

Comments should focus on high-level semantic explanations rather than low-level
syntactic details. For example, use "Compute the encodings" instead of "Doing a
concatenation followed by a linear projection, followed by a nonlinear
activation". The goal is to give a high-level overview of what a block of tensor
operations accomplishes.

When multiple tensor operations are chained, it is welcomed to add short inline
comments with the tensor shapes of computed tensors, e.g.:

```python
x = torch.cat([y, z], dim=1)  # (B, 2*C_in, H, W)
```

The symbols chosen in the comments should be consistent with the docstring
(possibly shortened versions of dimension names for explicitness).

**Rationale:**

High-level comments make complex tensor manipulation code more understandable
without cluttering it with excessive detail. Shape annotations help developers
track tensor dimensions through complex operations, catching shape mismatches
early. Consistency with docstring notation creates a unified mental model.

**Example:**

```python
def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Process input with context conditioning."""
    # Encode input features
    h = self.encoder(x)  # (B, C_enc, H, W)

    # Combine with context information
    c = self.context_proj(context)  # (B, C_enc)
    c = c[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])  # (B, C_enc, H, W)
    h = torch.cat([h, c], dim=1)  # (B, 2*C_enc, H, W)

    # Apply attention mechanism
    h = self.attention(h)  # (B, 2*C_enc, H, W)

    # Decode to output
    out = self.decoder(h)  # (B, C_out, H, W)

    return out
```

**Anti-pattern:**

```python
# WRONG: No comments
def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    h = self.encoder(x)
    c = self.context_proj(context)
    c = c[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
    h = torch.cat([h, c], dim=1)
    return self.decoder(self.attention(h))

# WRONG: Too low-level, syntactic comments
def forward(self, x, context):
    # Pass x through encoder layer
    h = self.encoder(x)
    # Project context using linear layer
    c = self.context_proj(context)
    # Add two None dimensions and expand
    c = c[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
```

---

### MOD-004: Model code is not self-contained

**Description:**

All utility functions for a model class should be organized together with the
model class in a clear and logical structure. Acceptable patterns include:

1. A single self-contained file: `physicsnemo/<models or nn>/model_name.py`
2. A subdirectory: `physicsnemo/<models or nn>/model_name/` containing:
   - `model_name.py` with the main model class
   - Additional modules for utility functions specific to this model

What should be avoided is a flat organization where model files and their
utility files are all mixed together in `physicsnemo/<models or nn>/`, making it
unclear which utilities belong to which models.

The only exception is when a utility function is used across multiple models. In
that case, the shared utility should be placed in an appropriate shared module.

**Rationale:**

Self-contained modules are easier to understand, maintain, and navigate. Having
all model-specific code in one place reduces cognitive load and makes it clear
which utilities are model-specific versus shared. This also simplifies code
reviews and reduces the likelihood of orphaned utility files when models are
refactored or removed.

**Example:**

```python
# Good Pattern 1: Single self-contained file
# File: physicsnemo/models/my_simple_model.py

def _compute_attention_mask(seq_length: int) -> torch.Tensor:
    """Helper function specific to MySimpleModel."""
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

class MySimpleModel(Module):
    """A simple model with utilities in same file."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = _compute_attention_mask(x.shape[1])
        return self._apply_attention(x, mask)

# Good Pattern 2: Subdirectory organization
# File: physicsnemo/models/my_complex_model/my_complex_model.py
from physicsnemo.models.my_complex_model.utils import helper_function

class MyComplexModel(Module):
    """A complex model with utilities in subdirectory."""
    pass

# File: physicsnemo/models/my_complex_model/utils.py
def helper_function(x):
    """Utility specific to MyComplexModel."""
    pass
```

**Anti-pattern:**

```python
# WRONG: Flat organization with utilities mixed in main directory
# File: physicsnemo/models/my_transformer.py
from physicsnemo.models.my_transformer_utils import _compute_mask  # WRONG

class MyTransformer(Module):
    pass

# File: physicsnemo/models/my_transformer_utils.py (WRONG: mixed with other models)
# File: physicsnemo/models/other_model.py
# File: physicsnemo/models/other_model_utils.py (WRONG: utilities scattered)
# All mixed together in flat structure - unclear organization!
```

---

### MOD-005: Invalid or missing tensor shape validation logic

**Description:**

All forward methods and other public methods that accept tensor arguments must
validate tensor shapes at the beginning of the method. This rule applies to:
- Individual tensor arguments
- Containers of tensors (lists, tuples, dictionaries)

For containers, validate their length, required keys, and the shapes of
contained tensors. Validation statements should be concise (ideally one check
per argument). Error messages must follow the standardized format:
`"Expected tensor of shape (B, D) but got tensor of shape {actual_shape}"`.

To avoid interactions with `torch.compile`, all validation must be wrapped in a
conditional check using `torch.compiler.is_compiling()`. Follow the "fail-fast"
approach by validating inputs before any computation.

**Rationale:**

Early shape validation catches errors at the API boundary with clear, actionable
error messages, making debugging significantly easier. Without validation, shape
mismatches result in cryptic errors deep in the computation graph. The
`torch.compile` guard ensures that validation overhead is eliminated in
production compiled code while preserving debug-time safety.

**Example:**

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Forward pass with shape validation."""
    ### Input validation
    # Skip validation when running under torch.compile for performance
    if not torch.compiler.is_compiling():
        # Extract expected dimensions
        B, C, H, W = x.shape if x.ndim == 4 else (None, None, None, None)

        # Validate x shape
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {C} channels"
            )

        # Validate optional mask
        if mask is not None:
            if mask.shape != (B, H, W):
                raise ValueError(
                    f"Expected mask shape ({B}, {H}, {W}), got {tuple(mask.shape)}"
                )

    # Actual computation happens after validation
    return self._process(x, mask)

def process_list(self, tensors: List[torch.Tensor]) -> torch.Tensor:
    """Process a list of tensors with validation."""
    ### Input validation
    if not torch.compiler.is_compiling():
        if len(tensors) == 0:
            raise ValueError("Expected non-empty list of tensors")

        # Validate all tensors have consistent shapes
        ref_shape = tensors[0].shape
        for i, t in enumerate(tensors[1:], start=1):
            if t.shape != ref_shape:
                raise ValueError(
                    f"All tensors must have the same shape. "
                    f"Tensor 0 has shape {tuple(ref_shape)}, "
                    f"but tensor {i} has shape {tuple(t.shape)}"
                )

    return torch.stack(tensors)
```

**Anti-pattern:**

```python
# WRONG: No validation at all
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layer(x)  # Will fail with cryptic error if shape is wrong

# WRONG: Validation not guarded by torch.compiler.is_compiling()
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:  # Breaks torch.compile
        raise ValueError(f"Expected 4D tensor, got {x.ndim}D")
    return self.layer(x)

# WRONG: Validation after computation has started
def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.layer1(x)  # Computation started
    if x.shape[1] != self.in_channels:  # Too late!
        raise ValueError(f"Wrong number of channels")
    return self.layer2(h)

# WRONG: Non-standard error message format
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if not torch.compiler.is_compiling():
        if x.ndim != 4:
            raise ValueError("Input must be 4D")  # Missing actual shape info
    return self.layer(x)
```

---

### MOD-006: Invalid or missing jaxtyping tensor annotations in public function signature

**Description:**

All tensor arguments and variables in model `__init__`, `forward`, and other
public methods must have type annotations using `jaxtyping`. This provides
runtime-checkable shape information in type hints.

Use the format `Float[torch.Tensor, "shape_spec"]` where shape_spec describes
tensor dimensions using space-separated dimension names (e.g., `"batch channels height width"`
or `"b c h w"`).

**Rationale:**

Jaxtyping annotations provide explicit, machine-readable documentation of
expected tensor shapes. This enables better IDE support, catches shape errors
earlier, and makes code more self-documenting. The annotations serve as both
documentation and optional runtime checks when jaxtyping's validation is
enabled.

**Example:**

```python
from jaxtyping import Float
import torch

class MyConvNet(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3)

    def forward(
        self,
        x: Float[torch.Tensor, "batch in_channels height width"]
    ) -> Float[torch.Tensor, "batch out_channels height width"]:
        """Process input with convolution."""
        return self.conv(x)

def process_attention(
    query: Float[torch.Tensor, "batch seq_len d_model"],
    key: Float[torch.Tensor, "batch seq_len d_model"],
    value: Float[torch.Tensor, "batch seq_len d_model"]
) -> Float[torch.Tensor, "batch seq_len d_model"]:
    """Compute attention with clear shape annotations."""
    pass
```

**Anti-pattern:**

```python
# WRONG: No jaxtyping annotations
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layer(x)

# WRONG: Using plain comments instead of jaxtyping
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch, channels, height, width)  # Use jaxtyping instead
    return self.layer(x)

# WRONG: Incomplete annotations (missing jaxtyping for tensor arguments)
def forward(
    self,
    x: Float[torch.Tensor, "b c h w"],
    mask: torch.Tensor  # Missing jaxtyping annotation
) -> Float[torch.Tensor, "b c h w"]:
    return self.layer(x, mask)
```

---

### MOD-007a: Cannot add required parameters without defaults

**Description:**

For any model in `physicsnemo/nn/module` or `physicsnemo/models`, adding new
required parameters (parameters without default values) to `__init__` or any
public method is strictly forbidden. This breaks backward compatibility.

New parameters must have default values to ensure existing code and checkpoints
continue to work. If a new parameter is truly required, increment the model
version number using `__model_checkpoint_version__` and add appropriate
versioning support.

**Rationale:**

Adding required parameters breaks all existing code that instantiates the model,
and breaks loading of old checkpoints. This violates PhysicsNeMo's commitment to
backward compatibility and would disrupt user workflows.

**Example:**

```python
# Good: Adding parameter with default value (backward compatible)
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"
    __supported_model_checkpoint_version__ = {
        "1.0": "Loading checkpoint from version 1.0 (current is 2.0). Still supported."
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,  # New parameter with default - backward compatible
        new_feature: bool = False  # New parameter with default - backward compatible
    ):
        super().__init__(meta=MyModelMetaData())
```

**Anti-pattern:**

```python
# WRONG: Adding required parameter without default
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        new_param: int  # WRONG: No default! Breaks old checkpoints
    ):
        super().__init__(meta=MyModelMetaData())
```

---

### MOD-007b: Cannot remove or rename parameters without compat mapper

**Description:**

For any model in `physicsnemo/nn/module` or `physicsnemo/models`, removing or
renaming parameters is strictly forbidden without proper backward compatibility
support.

If a parameter must be renamed or removed, the developer must:
1. Increment `__model_checkpoint_version__`
2. Add the old version to `__supported_model_checkpoint_version__` dict
3. Implement `_backward_compat_arg_mapper` classmethod to handle the mapping
4. Maintain support for the old API for at least 2 release cycles

**Rationale:**

Removing or renaming parameters breaks existing checkpoints and user code.
Proper version management and argument mapping ensures old checkpoints can still
be loaded and users have time to migrate to the new API.

**Example:**

```python
from typing import Any, Dict

# Good: Proper backward compatibility for parameter rename
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"
    __supported_model_checkpoint_version__ = {
        "1.0": (
            "Loading checkpoint from version 1.0 (current is 2.0). "
            "Parameter 'hidden_dim' renamed to 'hidden_size'."
        )
    }

    @classmethod
    def _backward_compat_arg_mapper(
        cls, version: str, args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map arguments from older versions."""
        args = super()._backward_compat_arg_mapper(version, args)

        if version == "1.0":
            # Map old parameter name to new name
            if "hidden_dim" in args:
                args["hidden_size"] = args.pop("hidden_dim")

            # Remove deprecated parameters
            if "legacy_param" in args:
                _ = args.pop("legacy_param")

        return args

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,  # Renamed from 'hidden_dim'
    ):
        super().__init__(meta=MyModelMetaData())
```

**Anti-pattern:**

```python
# WRONG: Renaming without backward compat
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"
    # Missing: __supported_model_checkpoint_version__ and _backward_compat_arg_mapper

    def __init__(self, input_dim: int, hidden_size: int):  # Renamed!
        super().__init__(meta=MyModelMetaData())
        # WRONG: Old checkpoints with 'hidden_dim' will fail!

# WRONG: Not calling super()
class MyModel(Module):
    @classmethod
    def _backward_compat_arg_mapper(cls, version: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # WRONG: Missing super()._backward_compat_arg_mapper(version, args)
        if "hidden_dim" in args:
            args["hidden_size"] = args.pop("hidden_dim")
        return args
```

---

### MOD-007c: Cannot change return types of public methods

**Description:**

For any model in `physicsnemo/nn/module` or `physicsnemo/models`, changing the return
type of any public method (including `forward`) is strictly forbidden. This
includes:
- Changing from returning a single value to returning a tuple
- Changing from a tuple to a single value
- Changing the number of elements in a returned tuple
- Changing the type of returned values

If a return type change is absolutely necessary, create a new method with a
different name and deprecate the old method following MOD-002b.

**Rationale:**

Changing return types is a breaking change that silently breaks user code. Users
who unpack return values or depend on specific return structures will experience
runtime errors. Unlike parameter changes (which can be managed with versioning),
return type changes affect runtime behavior and are harder to detect.

**Example:**

```python
# Good: Keeping consistent return type
class MyModel(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Always returns single tensor."""
        return self.process(x)

# Good: If new return is needed, add new method
class MyModel(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns output tensor."""
        output, loss = self._forward_with_loss(x)
        return output

    def forward_with_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """New method for returning both output and loss."""
        return self._forward_with_loss(x)
```

**Anti-pattern:**

```python
# WRONG: Changing return type
class MyModel(Module):
    __model_checkpoint_version__ = "2.0"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # WRONG: v1.0 returned single tensor, v2.0 returns tuple - breaks user code!
        return output, loss
```

---

### MOD-008a: Model missing constructor/attributes tests

**Description:**

Every model in `physicsnemo/nn/module` or `physicsnemo/models` must have tests that
verify model instantiation and all public attributes (excluding buffers and
parameters).

These tests should:
- Use `pytest` parameterization to test at least 2 configurations
- Test one configuration with all default arguments
- Test another configuration with non-default arguments
- Verify all public attributes have expected values

**Rationale:**

Constructor tests ensure the model can be instantiated correctly with various
configurations and that all attributes are properly initialized. This catches
issues early in the development cycle.

**Example:**

```python
@pytest.mark.parametrize(
    "config",
    ["default", "custom"],
    ids=["with_defaults", "with_custom_args"]
)
def test_my_model_constructor(config):
    """Test model constructor and attributes."""
    if config == "default":
        model = MyModel(input_dim=64, output_dim=32)
        assert model.hidden_dim == 128  # Default value
        assert model.dropout == 0.0  # Default value
    else:
        model = MyModel(
            input_dim=64,
            output_dim=32,
            hidden_dim=256,
            dropout=0.1
        )
        assert model.hidden_dim == 256
        assert model.dropout == 0.1

    # Test common attributes
    assert model.input_dim == 64
    assert model.output_dim == 32
```

**Anti-pattern:**

```python
# WRONG: Only testing default configuration
def test_my_model_bad():
    model = MyModel(input_dim=64, output_dim=32)
    # Only tests defaults
```

---

### MOD-008b: Model missing non-regression test with reference data

**Description:**

Every model must have non-regression tests that:
1. Instantiate the model with reproducible random parameters
2. Run forward pass with test data
3. Compare outputs against reference data saved in a `.pth` file

Requirements:
- Use `pytest` parameterization to test multiple configurations
- Test tensors must have realistic shapes (no singleton dimensions except batch)
- Test data should be meaningful and representative of actual use cases
- Compare actual tensor values, not just shapes
- All public methods (not just forward) need similar non-regression tests

**Critical:** Per MOD-002a, models cannot move out of experimental without these
tests.

**Rationale:**

Non-regression tests with reference data catch subtle numerical changes that
could break reproducibility. Simply checking output shapes is insufficient to
detect algorithmic changes or numerical instabilities.

**Example:**

```python
import pytest
import torch
from physicsnemo.models import MyModel

def _instantiate_model(cls, seed: int = 0, **kwargs):
    """Helper to create model with reproducible parameters."""
    model = cls(**kwargs)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=param.dtype))
    return model

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("config", ["default", "custom"])
def test_my_model_non_regression(device, config):
    """Test model forward pass against reference output."""
    if config == "default":
        model = _instantiate_model(MyModel, input_dim=64, output_dim=32)
    else:
        model = _instantiate_model(
            MyModel,
            input_dim=64,
            output_dim=32,
            hidden_dim=256
        )

    model = model.to(device)

    # Load reference data (meaningful shapes, no singleton dimensions)
    data = torch.load(f"test/models/data/my_model_{config}_v1.0.pth")
    x = data["x"].to(device)  # Shape: (4, 64), not (1, 64)
    out_ref = data["out"].to(device)

    # Run forward and compare values
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)
```

**Anti-pattern:**

```python
# WRONG: Only testing output shapes
def test_my_model_bad(device):
    model = MyModel(input_dim=64, output_dim=32).to(device)
    x = torch.randn(4, 64).to(device)
    out = model(x)
    assert out.shape == (4, 32)  # NOT SUFFICIENT!

# WRONG: Using singleton dimensions
def test_my_model_bad(device):
    x = torch.randn(1, 1, 64)  # WRONG: Trivial shapes
```

---

### MOD-008c: Model missing checkpoint loading test

**Description:**

Every model must have tests that load the model from a checkpoint file
(`.mdlus`) using `physicsnemo.Module.from_checkpoint()` and verify that:
1. The model loads successfully
2. All public attributes have expected values
3. Forward pass outputs match reference data

This ensures the model's serialization and deserialization work correctly.

**Critical:** Per MOD-002a, models cannot move out of experimental without these
tests.

**Rationale:**

Checkpoint tests verify that the model's custom serialization logic works
correctly and that saved models can be loaded in different environments. This is
critical for reproducibility and for users who need to save and load trained
models.

**Example:**

```python
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_my_model_from_checkpoint(device):
    """Test loading model from checkpoint and verify outputs."""
    model = physicsnemo.Module.from_checkpoint(
        "test/models/data/my_model_default_v1.0.mdlus"
    ).to(device)

    # Verify attributes after loading
    assert model.input_dim == 64
    assert model.output_dim == 32

    # Load reference data and verify outputs
    data = torch.load("test/models/data/my_model_default_v1.0.pth")
    x = data["x"].to(device)
    out_ref = data["out"].to(device)
    out = model(x)
    assert torch.allclose(out, out_ref, atol=1e-5, rtol=1e-5)
```

**Anti-pattern:**

```python
# WRONG: No checkpoint loading test
# (Missing test_my_model_from_checkpoint entirely)
```

---

### MOD-009: Avoid string-based class selection in model constructors

**Description:**

Passing a string that represents a class name, which is then used to instantiate
an internal submodule, should be avoided unless there are only a few choices (2
or 3 maximum) for the class name.

When there are more than 2-3 choices, the recommended practice is to pass an
already instantiated instance of a submodule instead of a string primitive for
dependency injection. This promotes better type safety, clearer APIs, and easier
testing.

**Rationale:**

String-based class selection makes code harder to type-check, debug, and test.
It obscures dependencies and makes it difficult for static analysis tools to
understand the code. Direct instance injection provides better IDE support,
type safety, and makes testing easier by allowing mock object injection.

**Example:**

```python
# Good: Limited choices (2-3 max) - string selection acceptable
class MyModel(Module):
    def __init__(
        self,
        activation: Literal["relu", "gelu"] = "relu"
    ):
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()

# Good: Many choices - use instance injection
class MyModel(Module):
    def __init__(
        self,
        encoder: Module,  # Pass instance, not string
        decoder: Module   # Pass instance, not string
    ):
        self.encoder = encoder
        self.decoder = decoder

# Usage:
model = MyModel(
    encoder=MyCustomEncoder(dim=128),
    decoder=MyCustomDecoder(dim=128)
)
```

**Anti-pattern:**

```python
# WRONG: String selection with many choices
class MyModel(Module):
    def __init__(
        self,
        encoder_type: str = "transformer"  # Many possible values
    ):
        # String-based factory pattern with 10+ choices
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder()
        elif encoder_type == "cnn":
            self.encoder = CNNEncoder()
        elif encoder_type == "rnn":
            self.encoder = RNNEncoder()
        # ... many more options
        # WRONG: Should accept encoder instance instead
```

---

### MOD-010: Avoid splatted kwargs in model constructors

**Description:**

Passing splatted arguments like `**kwargs_for_submodules` should be avoided in
model constructors as it might create conflicts in the names of these kwargs and
makes the API unclear.

Instead, it is recommended to pass non-splatted arguments in the form of a
`Dict` when configuration for submodules needs to be passed through. This makes
parameter passing explicit and avoids naming conflicts.

**Rationale:**

Splatted kwargs obscure the actual parameters being passed, make type checking
impossible, and can lead to subtle bugs from name conflicts. Explicit dictionary
parameters make the API clearer and enable better IDE support and error
detection.

**Example:**

```python
# Good: Explicit dict parameter
class MyModel(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_config: Optional[Dict[str, Any]] = None
    ):
        encoder_config = encoder_config or {}
        self.encoder = Encoder(input_dim=input_dim, **encoder_config)

# Usage:
model = MyModel(
    input_dim=64,
    output_dim=32,
    encoder_config={"hidden_dim": 128, "num_layers": 3}
)
```

**Anti-pattern:**

```python
# WRONG: Splatted kwargs
class MyModel(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **encoder_kwargs  # WRONG: Unclear what's accepted
    ):
        self.encoder = Encoder(input_dim=input_dim, **encoder_kwargs)
        # Risk of name conflicts, unclear API

# Usage - unclear what parameters are valid:
model = MyModel(input_dim=64, output_dim=32, hidden_dim=128, num_layers=3)
# Are hidden_dim and num_layers for MyModel or Encoder? Unclear!
```

---

### MOD-011: Use proper optional dependency handling

**Description:**

When a model requires optional dependencies (packages not installed by default),
use the PhysicsNeMo APIs for dependency handling:

1. **`check_min_version(package, version, hard_fail=False)`**: Use this function
   to check if a package is installed and available without actually importing
   it. Set `hard_fail=True` for hard requirements, `hard_fail=False` for soft
   requirements. This is the primary method for handling optional dependencies.

2. **`@require_version(package, version)`**: Use this decorator when core code
   must always be available but certain features need to be protected against
   older versions. This is rare and should only be used when you need to protect
   specific methods or classes against version incompatibilities.

3. **`pyproject.toml`**: This file is the one, only, and universal source of
   truth for all dependencies in PhysicsNeMo. All optional dependencies must be
   declared there.

**Rationale:**

Centralized dependency handling ensures consistent error messages and version
checking across the codebase. Checking availability without importing prevents
import errors and allows graceful degradation when optional packages are not
available. Using `pyproject.toml` as the single source of truth prevents
dependency specification from becoming scattered and inconsistent.

**Example:**

```python
import torch
from physicsnemo.core import Module
from physicsnemo.core.version_check import check_min_version, require_version

# Check optional dependency availability without importing
APEX_AVAILABLE = check_min_version("apex", "0.1.0", hard_fail=False)

class MyModel(Module):
    def __init__(
        self,
        input_dim: int,
        use_apex: bool = False
    ):
        super().__init__()
        self.use_apex = use_apex

        if use_apex and not APEX_AVAILABLE:
            raise RuntimeError(
                "apex is required for use_apex=True but is not installed. "
                "Install with: pip install apex>=0.1.0"
            )

        if use_apex:
            import apex  # Only import when actually needed
            self.fused_layer = apex.FusedLayer()
        else:
            self.fused_layer = None

# Using @require_version for protecting version-specific features
class AdvancedModel(Module):
    @require_version("torch", "2.4.0")
    def use_device_mesh(self):
        """This feature requires torch>=2.4.0."""
        from torch.distributed.device_mesh import DeviceMesh
        # Protected code that needs torch>=2.4.0
```

**Anti-pattern:**

```python
# WRONG: Direct import without checking availability
import apex  # Will fail if apex not installed!

class MyModel(Module):
    def __init__(self, use_apex: bool = False):
        if use_apex:
            self.layer = apex.FusedLayer()  # Already failed at import!

# WRONG: Try/except for dependency checking
try:
    import apex
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
# Use check_min_version instead!

# WRONG: Hardcoded version strings in multiple places
if version.parse(apex.__version__) < version.parse("0.1.0"):
    raise ImportError("apex>=0.1.0 required")
# Should use check_min_version or require_version!

# WRONG: Not declaring dependency in pyproject.toml
# All optional dependencies must be in pyproject.toml!
```

---

## Compliance

When implementing models, ensure all rules are followed. Code reviews should
verify each rule is followed and enforce the rules as strictly as possible.
For exceptions to these rules, document the reasoning in code comments and
obtain approval during code review.
