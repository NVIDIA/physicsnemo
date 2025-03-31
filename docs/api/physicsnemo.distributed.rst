PhysicsNeMo Distributed
========================

.. automodule:: physicsnemo.distributed
.. currentmodule:: physicsnemo.distributed

Distributed utilites in PhysicsNeMo are designed to simplify implementation of parallel training and
make inference scripts easier by providing a unified way to configure and query parameters associated 
with the distributed environment. The utilites in ``physicsnemo.distributed`` build on top of the 
utilites from ``torch.distributed`` and abstract out some of the complexities of setting up a
distributed execution environment.

The example below shows how to setup a simple distributed data parallel training recipe using the
distributed utilites in PhysicsNeMo. 
`DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ 
in PyTorch provides the framework for data parallel training by reducing parameter gradients
across multiple worker processes after the backwards pass. The code below shows how to specify 
the ``device_ids``, ``output_device``, ``broadcast_buffers`` and ``find_unused_parameters`` 
arguments of the ``DistributedDataParallel`` utility using the ``DistributedManager``. 

.. code:: python

    import torch
    from torch.nn.parallel import DistributedDataParallel
    from physicsnemo.distributed import DistributedManager
    from physicsnemo.models.mlp.fully_connected import FullyConnected

    def main():
        # Initialize the DistributedManager. This will automatically 
        # detect the number of processes the job was launched with and 
        # set those configuration parameters appropriately. Currently 
        # torchrun (or any other pytorch compatible launcher), mpirun (OpenMPI) 
        # and SLURM based launchers are supported.
        DistributedManager.initialize()

        # Since this is a singleton class, you can just get an instance 
        # of it anytime after initialization and not need to reinitialize
        # each time.
        dist = DistributedManager()

        # Set up model on the appropriate device. DistributedManager
        # figures out what device should be used on this process
        arch = FullyConnected(in_features=32, out_features=64).to(dist.device)

        # Set up DistributedDataParallel if using more than a single process.
        # The `distributed` property of DistributedManager can be used to 
        # check this.
        if dist.distributed:
            ddps = torch.cuda.Stream()
            with torch.cuda.stream(ddps):
                arch = DistributedDataParallel(
                    arch,
                    device_ids=[dist.local_rank],  # Set the device_id to be
                                                   # the local rank of this process on
                                                   # this node
                    output_device=dist.device,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters,
                )
            torch.cuda.current_stream().wait_stream(ddps)

        # Set up the optimizer
        optimizer = torch.optim.Adam(
            arch.parameters(),
            lr=0.001,
        )

        def training_step(input, target):
            pred = arch(invar)
            loss = torch.sum(torch.pow(pred - target, 2))
            loss.backward()
            optimizer.step()
            return loss

        # Sample training loop
        for i in range(20):
            # Random inputs and targets for simplicity
            input = torch.randn(128, 32, device=dist.device)
            target = torch.randn(128, 64, device=dist.device)

            # Training step
            loss = training_step(input, target)

    if __name__ == "__main__":
        main()

This training script can be run on a single GPU
using ``python train.py`` or on multiple GPUs using

.. code-block:: bash

   torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> train.py 

or 

.. code-block:: bash

   mpirun -np <num_gpus> python train.py 

if using OpenMPI. The script can also 
be run on a SLURM cluster using 

.. code-block:: bash 

   srun -n <num_gpus> python train.py

How does this work?
"""""""""""""""""""

An important aspect of the ``DistributedManager`` is that it is follows the 
`Borg pattern <https://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/>`_.
This means that ``DistributedManager`` essentially functions like a singleton 
class and once configured, all utilities in PhysicsNeMo can access the same configuration 
and adapt to the specified distributed structure.

For example, see the constructor of the ``DistributedAFNO`` class:

.. literalinclude:: ../../physicsnemo/models/afno/distributed/afno.py
   :pyobject: DistributedAFNO.__init__

This model parallel implementation can just instantiate ``DistributedManager`` and query 
if the process group named ``"model_parallel"`` exists and if so, what is it's size. Similarly, 
other utilities can query what device to run on, the total size of the distributed run, etc. 
without having to explicitly pass those params down the call stack.

.. note::

   This singleton/borg pattern is very useful for the ``DistributedManager`` since it takes charge 
   of bootstrapping the distributed run and unifies how all utilities become aware of the distributed 
   configuration. However, the singleton/borg pattern is not just a way to avoid passing parameters 
   to utilities. Use of this pattern should be limited and have good justification to avoid losing 
   tracability and keep the code readable.


.. autosummary::
   :toctree: generated

physicsnemo.distributed.manager
--------------------------------

.. automodule:: physicsnemo.distributed.manager
    :members:
    :show-inheritance:

physicsnemo.distributed.utils
-----------------------------

.. automodule:: physicsnemo.distributed.utils
    :members:
    :show-inheritance:

physicsnemo.distributed.autograd
--------------------------------

.. automodule:: physicsnemo.distributed.autograd
    :members:
    :show-inheritance:

physicsnemo.distributed.fft
----------------------------

.. automodule:: physicsnemo.distributed.fft
    :members:
    :show-inheritance:

physicsnemo.distributed.mappings
--------------------------------

.. automodule:: physicsnemo.distributed.mappings
    :members:
    :show-inheritance:
