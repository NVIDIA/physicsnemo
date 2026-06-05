<!-- markdownlint-disable -->
# Active Learning for Surface-CFD Aerodynamic Surrogates

This example lives under the CFD examples tree, alongside the surface-CFD
backbone it builds on:
[**`physicsnemo/examples/cfd/external_aerodynamics/active_learning_aero`**](../../cfd/external_aerodynamics/active_learning_aero/README.md).

It demonstrates end-to-end active learning on the
[ShiftSUV](https://huggingface.co/datasets/luminary-shift/SUV) surface-CFD
dataset using an uncertainty-aware GeoTransolver + Variational GP head, with
three plug-and-play acquisition strategies (UQ-driven, class-balanced random,
and latent-novelty). The AL loop itself is problem-agnostic — only the
physics/metrology hooks are CFD-specific — so the same recipe can drive any
uncertainty-quantified regression task.

See the [full README](../../cfd/external_aerodynamics/active_learning_aero/README.md)
for the recipe overview, configuration, results, and adapting-to-a-new-problem
guide.
