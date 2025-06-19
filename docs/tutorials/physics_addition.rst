Adding Physics-based Information
==================================

Adding inductive bias to the model training can be useful to improve the generalization
capability of the model. For Physics-AI, one way to do this is via designing neural
network architectures that are specifically designed to operate on data encountered in
this domain. FNOs, that use FFTs, Mesh Graph Networks that operate on directly on
simulation meshes, DoMINO model that operates on point clouds and has stencils similar
to some numerical methods are a great way to introduce inductive bias.

Additionally, one can also add the governing equations as loss functions to further
regularize the model predictions to obey the physics of the problem.

In this tutorial, we'll learn how to add physics-based loss terms to your model training
using PhysicsNeMo. `PhysicsNeMo-Sym <https://github.com/NVIDIA/physicsnemo-sym>`_
is a submodule of PhysicsNeMo that provides algorithms and utilities to physics-inform
the training of AI models. In this tutorial, we will explore the different utilities from
PhysicsNeMo-Sym, followed by sample end-to-end training workflows.

Adding Physics-based Losses
---------------------------------

Many AI models trained on physical data are designed to minimize the difference
between the model predictions and the true data. Typical loss functions used for this
purpose include MSE, RMSE, MAE, etc. Choosing the right loss function is an important
step for AI model training, and there is a lot of active research in this area. For
Physics-AI applications, we can take this idea even further and craft loss functions
that are suitable for the data. More specifically, the data used to train
models in the Physics-AI space typically comes from experimental measurements or from results
of a different numerical method. The system being studied is usually governed by physical laws
(such as conservation of mass, energy, etc.), and these methods or measurements
aim to satisfy these constraints. Adding these governing laws or equations as loss functions
can help the neural network satisfy these equations better and make the predictions more
physically interpretable.

Let's look at an example from the molecular dynamics domain.
Assume a system of molecules at equilibrium. Suppose we are training
a neural network to predict the forces on each molecule given the positions of the molecules.
Since the system is in equilibrium, we want to enforce that the total sum of all the
forces on each of the molecules is zero. This can be added as a constraint simply by doing the following:

.. code-block:: python

    ...
    # Assume a model forward pass
    # Model here is an mesh graph net and the system of molecules is represented
    # as a graph.
    # model outputs forces at each node (molecule) as a tensor of shape (N,3) 
    # where N is the number of molecules in the system
    out = model(node_features, edge_features, input_graph)
    loss_data = torch.nn.functional.l1_loss(out, true_out)                              # Regression / Data loss
    loss_physics = (1 / torch.shape(out)[0]) * torch.sum(torch.sum(out, dim=1)).abs()   # Sum of all forces, can also be written as torch.mean(out).abs()
    # define a lagrange multiplier / weight for the physics loss
    physics_weight = 0.001
    loss = loss_data + physics_weight * loss_physics
    loss.backward()
    optimizer.step()
    ...

Adding this, enforces an equilibrium condition on the model's prediction and is a
demonstration of how physics knowledge can be incorporated in your training workflow. 
A full example using this loss can be found in the `Molecular Dynamics Example <../examples/molecular_dynamics/lennard_jones/README.rst>`_

Adding PDE losses
~~~~~~~~~~~~~~~~~~~~~~~

Sometimes, the governing equations for the problem can include Partial Differential Equations.
This is especially true for problems in physics and other scientific domains. Computing local residuals
of these PDEs on a model predictions involves computing spatial and temporal gradients and the methods
used can vary based on the model architecture. 

PhysicsNeMo-Sym aims to simplify this process. 
The `PhysicsInformer <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#module-physicsnemo.sym.eq.phy_informer>`_
utility can compute the PDE losses on point clouds, grids, or graphs/meshes using
techniques such as Automatic Differentiation, Finite Difference, Meshless Finite Difference,
Least Squares, Spectral Differentiation, etc. 

The ``PhysicsInformer`` class requires equations to compute as a parameter. The equation
must be in the form of PhysicsNeMo-Sym's PDE. Custom PDEs are also supported. 

Some general comments for the ``PhysicsInformer``:


- Using ``PhysicsInformer``, you can physics-inform almost any model architecture
  (point-cloud based, grid based, graph based, etc.), e.g. MLPs, DeepONets, FNOs, CNNs, Diffusion Models,
  Graph networks, etc.
- The utility supports gradients via Automatic Differentiation, Spectral,
  Finite Difference, Meshless Finite Difference and Least Squares methods. (See following section for more info)
- Given the PDE governing equations and ``required_outputs``, this utility constructs the computational graph,
  and computes the gradients efficiently, to output the residuals.
- This utility simplifies the spatial derivative computation. 
  If using transient PDEs, the temporal derivative will have to be computed separately
  and passed as an input to the ``forward()`` call.
  Refer `API Doc <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#module-physicsnemo.sym.eq.phy_informer>`_ for more details.
- Different spatial gradient computing methods require different inputs to the ``forward()`` call. 
  To identify the inputs that need to be passed to the ``forward()`` call, you 
  can access the value of ``.required_inputs`` property.


The various spatial derivative methods and their applicability based on the model output type can be summarized below

- `Automatic Differentiation <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#physicsnemo.sym.eq.spatial_grads.spatial_grads.GradientsAutoDiff>`_: Suitable for outputs from models which are differentiable w.r.t model inputs.
  Model inputs must include coordinates as inputs. Ideal for MLP type of architectures, and can operate on point clouds,
  structured grids, or unstructured meshes as long as the differentiability and input constraints are satisfied.
  Computationally expensive but more accurate than other numerical gradient methods.

- `Meshless Finite Difference <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#physicsnemo.sym.eq.spatial_grads.spatial_grads.GradientsMeshlessFiniteDifference>`_: Numerical gradient method suitable for models that can
  predict field values on stencil points in addition to the original points. The points can come from either a point cloud or 
  grid (structured or unstructured); point clouds are most suitable applications. 
  Fast, but can present numerical instability if ``fd_dx`` is too small.

- `Finite Difference <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#physicsnemo.sym.eq.spatial_grads.spatial_grads.GradientsFiniteDifference>`_: Numerical gradient method suitable for models that output predictions on
  structured grids with uniform spacing (each dimension can have a different spacing of its own).

- `Spectral Derivatives <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#physicsnemo.sym.eq.spatial_grads.spatial_grads.GradientsSpectral>`_: Numerical gradient method suitable for models that output predictions
  on structured grids with uniform spacing (each dimension can have a different spacing of its own) and have
  periodic boundaries.

- `Least Squares Method <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#physicsnemo.sym.eq.spatial_grads.spatial_grads.GradientsLeastSquares>`_: Numerical gradient method most suitable for models predict output on unstructured
  grids or structured grids with non-uniform spacing. 


Computing PDE losses using Automatic Differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example of using the ``autodiff`` method to compute the residuals.
A few things to note when using the ``autodiff`` method:

- Ensure the model is differentiable enough for the PDE being used. 
  - C\ :sup:`1`\ Continuous for a First-Order PDE
  - C\ :sup:`2`\ Continuous for a Second-Order PDE
  - ...
- E.g. a model that uses ReLU activation function will have it's second derivatives zero.
  So using automatic differentiation based gradients is not recommended. 
- For all spatial coordinate tensors (e.g., `x`, `y`, and `z`), call the method ``x.requires_grad_(True)`` to enable gradient tracking.
- Coordinates is a tensor of shape ``(N, D)`` shaped tensor, where ``D`` is the number of spatial dimensions.
- This method is accurate but more computationally expensive compared to some 
  other numerical methods due to automatic differentiation. 

.. code-block:: python

    import torch
    import numpy as np
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


    class Model(torch.nn.Module):
        """Define a dummy model"""
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x_input):
            x, y, z = x_input[:, 0:1], x_input[:, 1:2], x_input[:, 2:3]
            
            # compute u, v, w, p
            u = x * y * z
            v = x * y ** 2 * z
            w = x ** 2 * y * z
            p = x * y * z ** 2

            return torch.cat([u, v, w, p], dim=1)

    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)  # requires_grad_ is set to True to enable Automatic Differentiation
    y = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    z = torch.linspace(0, 2 * np.pi, steps=steps).requires_grad_(True)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # instantiate model
    model = Model()

    # use the Navier Stokes PDE from Sym's PDE module
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)   # Coords shape: (1000000, 3)
    
    # instantiate PhysicsInformer with autodiff method.
    # choosing NavierStokes PDE will enable us to query continuity, and momentum in x, y, z directions
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="autodiff",
        device=coords.device,
    )

    # model forward pass
    # this needs to be differentiable as explained above for auto-diff gradients to work
    # if the model does not satisfy these requirements, follow along this tutorial to
    # see numerical ways to compute the derivatives.
    out = model(coords)

    # compute the residuals
    # this returns a dict containing tensors for required_outputs
    residuals = phy_informer.forward(
        {
            "coordinates": coords,
            "u": out[:, 0:1],
            "v": out[:, 1:2],
            "w": out[:, 2:3],
            "p": out[:, 3:4],            
        },
    )

A full example using this loss can be found in the `Physics Informed Darcy Flow Example <../examples/cfd/darcy_physics_informed/README.rst>`_

Computing PDE losses using Mesh-less Finite Difference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example of using the ``meshless_finite_difference`` method to compute the residuals.
A few things to note when using the ``meshless_finite_difference`` method:

- In Addition to the outputs at the original data points, outputs are needed on the 
  stencil points. The stencil points can be computed using 
  ``physicsnemo.sym.eq.spatial_grads.spatial_grads.compute_stencil3d`` function
  from PhysicsNeMo Sym. Stencil points are defined using the following convention
  "u>>x::1": u(i+1, j) "u>>x::-1": u(i-1, j) "u>>x::1&&y::1": u(i+1, j+1) "u>>x::-1&&y::-1": u(i-1, j-1) etc.
  To identify the inputs that need to be passed to the ``forward()`` call, you 
  can access the value of ``.required_inputs`` property.
- ``fd_dx`` is a hyperparameter. Smaller value typically yields more accurate
  gradients, but can lead to numerical instability. A value of 0.001 is a good
  value to start, assuming the variation of spatial coordinates in the problem is $\mathcal{O}(1)$.

.. code-block:: python

    import torch
    import numpy as np
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


    class Model(torch.nn.Module):
        """Define a dummy model"""
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x_input):
            x, y, z = x_input[:, 0:1], x_input[:, 1:2], x_input[:, 2:3]
            
            # compute u, v, w, p
            u = x * y * z
            v = x * y ** 2 * z
            w = x ** 2 * y * z
            p = x * y * z ** 2

            return torch.cat([u, v, w, p], dim=1)

    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps)
    y = torch.linspace(0, 2 * np.pi, steps=steps)
    z = torch.linspace(0, 2 * np.pi, steps=steps)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # instantiate model
    model = Model()

    # use the Navier Stokes PDE from Sym's PDE module
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)   # Coords shape: (1000000, 3)
    
    # instantiate PhysicsInformer with meshless_finite_difference method.
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="meshless_finite_difference",
        fd_dx=0.001,
        device=coords.device,
    )

    # model forward pass
    out = model(coords)

    # Compute stencil points and their forward pass
    po_posx, po_negx, po_posy, po_negy, po_posz, po_negz = compute_stencil3d(
        coords, model, dx=0.001
    )

    # compute the residuals
    # pass all the variables computed on stencil points
    # this returns a dict containing tensors for required_outputs
    residuals = phy_informer.forward(
        {
            "u": out[:, 0:1],
            "v": out[:, 1:2],
            "w": out[:, 2:3],
            "p": out[:, 3:4],
            "u>>x::1": po_posx[:, 0:1],
            "v>>x::1": po_posx[:, 1:2],
            "w>>x::1": po_posx[:, 2:3],
            "p>>x::1": po_posx[:, 3:4],
            "u>>x::-1": po_negx[:, 0:1],
            "v>>x::-1": po_negx[:, 1:2],
            "w>>x::-1": po_negx[:, 2:3],
            "p>>x::-1": po_negx[:, 3:4],
            "u>>y::1": po_posy[:, 0:1],
            "v>>y::1": po_posy[:, 1:2],
            "w>>y::1": po_posy[:, 2:3],
            "p>>y::1": po_posy[:, 3:4],
            "u>>y::-1": po_negy[:, 0:1],
            "v>>y::-1": po_negy[:, 1:2],
            "w>>y::-1": po_negy[:, 2:3],
            "p>>y::-1": po_negy[:, 3:4],
            "u>>z::1": po_posz[:, 0:1],
            "v>>z::1": po_posz[:, 1:2],
            "w>>z::1": po_posz[:, 2:3],
            "p>>z::1": po_posz[:, 3:4],
            "u>>z::-1": po_negz[:, 0:1],
            "v>>z::-1": po_negz[:, 1:2],
            "w>>z::-1": po_negz[:, 2:3],
            "p>>z::-1": po_negz[:, 3:4],      
        },
    )

Computing PDE losses using Finite Difference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example of using the ``finite_difference`` method to compute the residuals.
A few things to note when using the ``finite_difference`` method:

- This method uses the second-order central finite difference scheme to compute the
  gradients on a structured grid.
- ``fd_dx`` parameter is based on the grid spacing of the input.

.. code-block:: python

    import torch
    import numpy as np
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


    class Model(torch.nn.Module):
        """Define a dummy model"""
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x_input):
            x, y, z = x_input[:, 0:1], x_input[:, 1:2], x_input[:, 2:3]
            
            # compute u, v, w, p
            u = x * y * z
            v = x * y ** 2 * z
            w = x ** 2 * y * z
            p = x * y * z ** 2

            return torch.cat([u, v, w, p], dim=1)

    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps)
    y = torch.linspace(0, 2 * np.pi, steps=steps)
    z = torch.linspace(0, 2 * np.pi, steps=steps)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # instantiate model
    model = Model()

    # use the Navier Stokes PDE from Sym's PDE module
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # Coords shape: (1, 3, 100, 100, 100)
    
    # instantiate PhysicsInformer with finite_difference method.
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="finite_difference",
        fd_dx=(2 * np.pi / steps),  # computed based on the grid spacing
        device=coords.device,
    )

    # model forward pass
    out = model(coords)

    # compute the residuals
    # this returns a dict containing tensors for required_outputs
    residuals = phy_informer.forward(
        {
            "u": out[:, 0:1],
            "v": out[:, 1:2],
            "w": out[:, 2:3],
            "p": out[:, 3:4],
        },
    )

A full example using this loss can be found in the `Physics Informed Darcy Flow Example <../examples/cfd/darcy_physics_informed/README.rst>`_

Computing PDE losses using Spectral Derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example of using the ``spectral`` method to compute the residuals.
A few things to note when using the ``spectral`` method:

- This method works well for periodic domains, while for non-periodic domains, it 
  is known to produce artifacts at the boundaries. Appropriate padding is required. 
- ``bounds`` parameter is based on the size of the domain.

.. code-block:: python

    import torch
    import numpy as np
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


    class Model(torch.nn.Module):
        """Define a dummy model"""
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x_input):
            x, y, z = x_input[:, 0:1], x_input[:, 1:2], x_input[:, 2:3]
            
            # compute u, v, w, p
            u = x * y * z
            v = x * y ** 2 * z
            w = x ** 2 * y * z
            p = x * y * z ** 2

            return torch.cat([u, v, w, p], dim=1)

    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps)
    y = torch.linspace(0, 2 * np.pi, steps=steps)
    z = torch.linspace(0, 2 * np.pi, steps=steps)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # instantiate model
    model = Model()

    # use the Navier Stokes PDE from Sym'ss PDE module
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    coords = torch.stack([xx, yy, zz], dim=0).unsqueeze(0)  # Coords shape: (1, 3, 100, 100, 100)
    
    # instantiate PhysicsInformer with spectral method.
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="spectral",
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
        device=coords.device,
    )

    # model forward pass
    out = model(coords)

    # compute the residuals
    # this returns a dict containing tensors for required_outputs
    residuals = phy_informer.forward(
        {
            "u": out[:, 0:1],
            "v": out[:, 1:2],
            "w": out[:, 2:3],
            "p": out[:, 3:4],
        },
    )

Computing PDE losses using Least-Squares Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code below shows an example of using the ``least_squares`` method to compute the residuals.
A few things to note when using the ``least_squares`` method:

- This method is designed to compute gradients for unstructured meshes / grids.
- All gradient and residual quantities are computed on the node points. 
- This method also requires connectivity information, which can typically be pre-computed.
  Alternatively, you can also use ``physicsnemo.sym.eq.spatial_grads.spatial_grads.compute_connectivity_tensor``
  function to compute the connectivity tensor. 

.. code-block:: python

    import torch
    import numpy as np
    from physicsnemo.sym.eq.phy_informer import PhysicsInformer
    from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes


    class Model(torch.nn.Module):
        """Define a dummy model"""
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, x_input):
            x, y, z = x_input[:, 0:1], x_input[:, 1:2], x_input[:, 2:3]
            
            # compute u, v, w, p
            u = x * y * z
            v = x * y ** 2 * z
            w = x ** 2 * y * z
            p = x * y * z ** 2

            return torch.cat([u, v, w, p], dim=1)

    steps = 100
    x = torch.linspace(0, 2 * np.pi, steps=steps)
    y = torch.linspace(0, 2 * np.pi, steps=steps)
    z = torch.linspace(0, 2 * np.pi, steps=steps)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")

    # instantiate model
    model = Model()

    # use the Navier Stokes PDE from Sym's PDE module
    ns = NavierStokes(nu=0.01, rho=1.0, dim=3, time=False)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)  # Coords shape: (1000000, 3)

    # Sample code to compute node ids and edges. This information is typically
    # available from the mesh / graph representation. 
    edge_ids = []
    if steps > 1:
        # Edges in the i-direction
        edges_i = torch.stack([index[: -steps * steps], index[steps * steps :]], dim=1)
        edge_ids.append(edges_i)

        # Edges in the j-direction
        edges_j = torch.stack([index[:-steps], index[steps:]], dim=1)
        edge_ids.append(edges_j)

        # Edges in the k-direction
        edges_k = torch.stack([index[:-1], index[1:]], dim=1)
        edge_ids.append(edges_k)

    edge_ids = torch.cat(edge_ids).to(device)

    node_ids = torch.arange(coords_unstructured.size(0)).reshape(-1, 1).to(device)

    # instantiate PhysicsInformer with least_squares method.
    phy_informer = PhysicsInformer(
        required_outputs=["continuity", "momentum_x"],
        equations=ns,
        grad_method="least_squares",
        bounds=[2 * np.pi, 2 * np.pi, 2 * np.pi],
        device=coords.device,
        compute_connectivity=True   # Compute connectivity using the node and edge information
    )

    # model forward pass
    out = model(coords)

    # compute the residuals
    # pass the connectivity information
    # this returns a dict containing tensors for required_outputs
    residuals = phy_informer.forward(
        {
            "coordinates": coords,
            "nodes": node_ids,  # can be obtained from the graph representation, eg. graph.nodes() 
            "edges": edge_ids,  # can be obtained from the graph representation, eg. graph.edges()
            "u": out[:, 0:1],
            "v": out[:, 1:2],
            "w": out[:, 2:3],
            "p": out[:, 3:4],
        },
    )

A full example using this loss can be found in the `Stokes Flow Example <../examples/cfd/stokes_mgn/README.rst>`_

Customizing the PDEs
~~~~~~~~~~~~~~~~~~~~~~

PhysicsNeMo Sym's symbolic library, 
allows users to define the equations using SymPy.
PhysicsNeMo Sym comes with several built-in PDEs that are customizable such that they can
be applied to steady-state or transient problems in 1D/2D/3D
(this is not applicable to all PDEs). 
A non-exhaustive list of PDEs that are currently available in PhysicsNeMo Sym include:

- AdvectionDiffusion: Advection diffusion equation
- GradNormal: Normal gradient of a scalar
- Diffusion: Diffusion equation
- MaxwellFreqReal: Frequency domain Maxwell's equation
- LinearElasticity: Linear elasticity equations
- LinearElasticityPlaneStress: Linear elasticity plane stress equations
- NavierStokes: Navier stokes equations for fluid flow
- ZeroEquation: Zero equation turbulence model
- WaveEquation: Wave equation

For a tutorial on writing custom PDEs, refer `Custom PDEs <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/nodes.html#custom-pdes>`_.

Using the gradients directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you only need access to spatial gradients without the need to compute the residuals, 
you can use the `GradientCalculator <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.eq.html#module-physicsnemo.sym.eq.spatial_grads.spatial_grads>`_
directly. Refer to the API docs for more details.

Using geometry information
------------------------------

PhysicsNeMo also provides several ways to incorporate geometry information into the training pipelines.
From computing signed distance fields for implicit geometry representation, to sampling point-clouds,
utilities from PhysicsNeMo and more specifically PhysicsNeMo-Sym can be used to enrich the model training
using geometry information.

Below is a non-exhaustive list of different ways geometry information, derived from PhysicsNeMo can be
used:

- Compute point-clouds for training and inference
- Compute implicit geometry representation using Signed Distance Fields, which can be used
  to train surrogate models in the absence of / addition to mesh information
- Apply boundary conditions
- ...

Let's review some of these below.

Computing Signed Distance Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mathematically, signed distance field or signed distance function (SDF) is defined as the orthogonal distance
of a given point to the nearest boundary / surface of a geometric shape. It is widely used to describe the geometry
in mathematics, rendering, and similar applications. In physics-informed learning, it is also used to represent as
`geometric inputs to neural networks <https://www.research.autodesk.com/app/uploads/2023/03/convolutional-neural-networks-for.pdf_rectr0tDKzFYVAAJe.pdf>`_.

Inside PhysicsNeMo, there are several ways to compute the SDF of a geometry. 

- Using the ``physicsnemo.utils.sdf.signed_distance_field``:
  
  This function is useful for computing SDF from a given mesh and input points.
  The code below gives a sample implementation

  .. code-block:: python

    import pyvista as pv
    import numpy as np
    from physicsnemo.utils.sdf import signed_distance_field

    # Download the Stanford Bunny STL from https://commons.wikimedia.org/wiki/File:Stanford_Bunny.stl
    mesh = pv.read("Stanford_Bunny.stl")
    faces = mesh.faces.reshape((-1, 4))
    mesh_vertices = [tuple(face[1:]) for face in faces]
    mesh_indices = np.arange(0, mesh.points.shape[0])

    # Compute the signed distance field at the (0, 0, 0)
    signed_distance_field(mesh_vertices, mesh_indices, (0, 0, 0))

- Using the ``.sdf`` attribute of the ``Tessellation`` module from PhysicsNeMo Sym:

  PhysicsNeMo Sym allows you to load STL files and also define custom geometries
  using Constructive Solid Geometry and use it for computing the SDF. The 
  `geometry module documentation from PhysicsNeMo Sym <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/csg_and_tessellated_module.html#>`_
  provides a comprehensive documentation of this functionality. The code below shows
  a sample implementation of this

  .. code-block:: python

    import numpy as np
    from physicsnemo.sym.geometry.tessellation import Tessellation

    # read the Stanford Bunny stl
    geo = Tessellation.from_stl("./Stanford_Bunny.stl")

    # compute the SDF on the (0, 0, 0) points
    sdf = geo.sdf(
            {
                "x": np.array([[0]]),   # each coordinate must have shape (N, 1)
                "y": np.array([[0]]),
                "z": np.array([[0]]),
            },
        params={}
    )["sdf"]

A few examples using SDF during training / inference can be found in the 
`External Aerodynamics using DoMINO Example <../examples/cfd/external_aerodynamics/domino/README.rst>`_, 
`Datacenter CFD example <../examples/cfd/datacenter/README.rst>`_.

Sampling Point Clouds
~~~~~~~~~~~~~~~~~~~~~~~

The `geometry module from PhysicsNeMo Sym <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/features/csg_and_tessellated_module.html#>`_
also allows sampling of uniform point clouds in the interior (volume) and surface
of the geometry. The sampled point clouds can be used to apply a variety of physics constraints
during training, for example boundary conditions or even used during model inference
to bypass the need for mesh generation.

The ``sample_interior()`` and ``sample_boundary()`` methods can be used on the geometry objects
to sample the points in the interior and on the surface respectively. Please refer Sym's docs for
more details. 

This capability can be further extended to form a geometry datapipe. For example,
one can create a datapipe to sample points on the surface of multiple STLs or
multiple CSG type of geometries. You can use the ``GeometryDatapipe`` from PhysicsNeMo Sym
for this purpose. Refer API docs for `GeometryDatapipe <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/api/physicsnemo.sym.geometry.html#module-physicsnemo.sym.geometry.geometry_dataloader>`_ for more details. 

The code below shows a sample datapipe.

.. code-block:: python

    from physicsnemo.sym.geometry.geometry_dataloader import GeometryDatapipe
    from physicsnemo.sym.geometry.tessellation import Tessellation

    geoms = []
    # We will just create a datapipe of 10 same Stanford Bunny geometries
    for i in range(10):
        geo = Tessellation.from_stl("./Stanford_bunny.stl")
        geoms.append(geo)
    
    datapipe = GeometryDatapipe(
        geom_objects=geoms,
        sample_type="surface",
        num_points=100,
        batch_size=2,
        num_workers=1,
        device="cuda",
    )

    for data in datapipe:
        print(data[0].keys()) # For surface sampling, this should print ["x", "y", "z", "area", "normal_x", "normal_y", "normal_z"]

A full example using this for boundary and interior sampling can be found in the `Lid Driven Cavity Flow Example <../examples/cfd/ldc_pinns/README.rst>`_.
Furthermore, several examples from `PhysicsNeMo Sym <https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/index.html>`_ leverage
similar functionality to solve a variety of problems using PINNs. 
