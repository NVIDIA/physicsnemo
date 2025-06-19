NVIDIA PhysicsNeMo Core (Latest Release)
========================================

NVIDIA PhysicsNeMo is an open-source deep-learning framework for building, training,
fine-tuning and inferring Physics AI models using state-of-the-art SciML methods
for AI4science and engineering.

PhysicsNeMo provides python modules to compose scalable and optimized training and
inference pipelines to explore, develop, validate and deploy AI models that combine
physics knowledge with data, enabling real-time predictions.

Whether you are exploring the use of Neural operators, GNNs, or transformers or are
interested in Physics-informed Neural Networks or a hybrid approach in between,
PhysicsNeMo provides you with an optimized stack that will enable you to train your
models at scale.

.. figure:: /img/value_prop/Knowledge_guided_models.gif
   :alt: PhysicsNeMo Value Prop
   :width: 80.0%
   :align: center


.. toctree::
   :maxdepth: 2
   :caption: PhysicsNeMo User Guide
   :name: PhysicsNeMo User Guide

   tutorials/simple_training_example.rst
   tutorials/simple_logging_and_checkpointing.rst
   tutorials/profiling.rst
   tutorials/performance.rst
   tutorials/domain_parallelism_entry_point.rst
   tutorials/physics_addition.rst

.. toctree::
   :maxdepth: 2
   :caption: PhysicsNeMo API
   :name: PhysicsNeMo API

   api/physicsnemo.models.rst
   api/physicsnemo.datapipes.rst
   api/physicsnemo.metrics.rst
   api/physicsnemo.deploy.rst
   api/physicsnemo.distributed.rst
   api/physicsnemo.distributed.shardtensor.rst
   api/physicsnemo.utils.rst
   api/physicsnemo.launch.logging.rst
   api/physicsnemo.launch.utils.rst

.. toctree::
   :maxdepth: 1
   :caption: Introductory examples for learning key ideas
   :name: Introductory examples for learning key ideas

   examples/cfd/darcy_fno/README.rst
   examples/cfd/darcy_physics_informed/README.rst
   examples/cfd/ldc_pinns/README.rst
   examples/cfd/vortex_shedding_mgn/README.rst
   examples/weather/fcn_afno/README.rst
   examples/cfd/lagrangian_mgn/README.rst
   examples/cfd/stokes_mgn/README.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples: CFD
   :name: Examples: CFD

   examples/cfd/vortex_shedding_mgn/README.rst
   examples/cfd/external_aerodynamics/aero_graph_net/README.rst
   examples/cfd/external_aerodynamics/domino/README.rst
   examples/cfd/external_aerodynamics/figconvnet/README.rst
   examples/cfd/external_aerodynamics/xaeronet/README.rst
   examples/cfd/navier_stokes_rnn/README.rst
   examples/cfd/gray_scott_rnn/README.rst
   examples/cfd/lagrangian_mgn/README.rst
   examples/cfd/darcy_nested_fnos/README.rst
   examples/cfd/darcy_physics_informed/README.rst
   examples/cfd/stokes_mgn/README.rst
   examples/cfd/lid_driven_cavity/README.rst
   examples/cfd/swe_distributed_gnn/README.rst
   examples/cfd/vortex_shedding_mesh_reduced/README.rst
   examples/cfd/darcy_transolver/README.rst
   examples/cfd/flow_reconstruction_diffusion/README.rst
   examples/cfd/datacenter/README.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples: Weather and Climate
   :name: Examples: Weather and Climate

   examples/weather/dataset_download/README.rst
   examples/weather/graphcast/README.rst
   examples/weather/fcn_afno/README.rst
   examples/weather/dlwp/README.rst
   examples/weather/dlwp_healpix/README.rst
   examples/weather/diagnostic/README.rst
   examples/weather/unified_recipe/README.rst
   examples/weather/corrdiff/README.rst
   examples/weather/stormcast/README.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples: Healthcare
   :name: Examples: Healthcare

   examples/healthcare/bloodflow_1d_mgn/README.rst
   examples/healthcare/brain_anomaly_detection/README.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples: Additive Manufacturing
   :name: Examples: Additive Manufacturing

   examples/additive_manufacturing/sintering_physics/README.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples: Molecular Dynamics
   :name: Examples: Molecular Dynamics

   examples/molecular_dynamics/lennard_jones/README.rst
