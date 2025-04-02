# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import os
import queue
import warnings
from typing import Optional, Tuple
from warnings import warn

import numpy as np
import torch
import torch.distributed as dist

from physicsnemo.distributed.config import ProcessGroupConfig, ProcessGroupNode
from physicsnemo.utils.version_check import check_min_version, require_version

# warnings.simplefilter("default", DeprecationWarning)


class PhysicsNeMoUndefinedGroupError(Exception):
    """Exception for querying an undefined process group using the PhysicsNeMo DistributedManager"""

    def __init__(self, name: str):
        """

        Parameters
        ----------
        name : str
            Name of the process group being queried.

        """
        message = (
            f"Cannot query process group '{name}' before it is explicitly created."
        )
        super().__init__(message)


class PhysicsNeMoUninitializedDistributedManagerWarning(Warning):
    """Warning to indicate usage of an uninitialized DistributedManager"""

    def __init__(self):
        message = (
            "A DistributedManager object is being instantiated before "
            + "this singleton class has been initialized. Instantiating a manager before "
            + "initialization can lead to unexpected results where processes fail "
            + "to communicate. Initialize the distributed manager via "
            + "DistributedManager.initialize() before instantiating."
        )
        super().__init__(message)


class DistributedManager(object):
    """Distributed Manager for setting up distributed training environment.

    This is a singleton that creates a persistance class instance for storing parallel
    environment information through out the life time of the program. This should be
    used to help set up Distributed Data Parallel and parallel datapipes.

    Note
    ----
    One should call `DistributedManager.initialize()` prior to constructing a manager
    object

    Example
    -------
    >>> DistributedManager.initialize()
    >>> manager = DistributedManager()
    >>> manager.rank
    0
    >>> manager.world_size
    1
    """

    _shared_state = {}

    def __new__(cls):
        obj = super(DistributedManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_rank"):
            obj._rank = 0
        if not hasattr(obj, "_world_size"):
            obj._world_size = 1
        if not hasattr(obj, "_local_rank"):
            obj._local_rank = 0
        if not hasattr(obj, "_distributed"):
            obj._distributed = False
        if not hasattr(obj, "_device"):
            obj._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not hasattr(obj, "_cuda"):
            obj._cuda = torch.cuda.is_available()
        if not hasattr(obj, "_broadcast_buffers"):
            obj._broadcast_buffers = False
        if not hasattr(obj, "_find_unused_parameters"):
            obj._find_unused_parameters = False
        if not hasattr(obj, "_initialization_method"):
            obj._initialization_method = "None"
        if not hasattr(obj, "_groups"):
            obj._groups = {}
        if not hasattr(obj, "_group_ranks"):
            obj._group_ranks = {}
        if not hasattr(obj, "_group_names"):
            obj._group_names = {}
        if not hasattr(obj, "_is_initialized"):
            obj._is_initialized = False
        if not hasattr(obj, "_global_mesh"):
            obj._global_mesh = None  # Lazy initialized right when it's first needed
        if not hasattr(obj, "_mesh_dims"):
            obj._mesh_dims = {}  # Dictionary mapping axis names to sizes

        return obj

    def __init__(self):
        if not self._is_initialized:
            raise PhysicsNeMoUninitializedDistributedManagerWarning()
        super().__init__()

    @property
    def rank(self):
        """Process rank"""
        return self._rank

    @property
    def local_rank(self):
        """Process rank on local machine"""
        return self._local_rank

    @property
    def world_size(self):
        """Number of processes in distributed environment"""
        return self._world_size

    @property
    def device(self):
        """Process device"""
        return self._device

    @property
    def distributed(self):
        """Distributed environment"""
        return self._distributed

    @property
    def cuda(self):
        """If cuda is available"""
        return self._cuda

    @property
    def mesh_dims(self):
        """Mesh Dimensions as dictionary (axis name : size)"""
        return self._mesh_dims

    @property
    def group_names(self):
        """
        Returns a list of all named process groups created
        """
        return self._groups.keys()

    @property
    def global_mesh(self):
        """
        Returns the global mesh.  If it's not initialized, it will be created when this is called.
        """

        # Properties don't mesh with decorators.  So in this function, I call the check manually:
        check_min_version("torch", "2.4")

        if self._global_mesh is None:
            # Fully flat mesh (1D) by default:
            self.initialize_mesh(mesh_shape=(-1,), mesh_dim_names=("world",))

        return self._global_mesh

    @require_version("torch", "2.4")
    def mesh_names(self):
        """
        Return mesh axis names
        """
        return self._mesh_dims.keys()

    @require_version("torch", "2.4")
    def mesh_sizes(self):
        """
        Return mesh axis sizes
        """
        return self._mesh_dims.values()

    def group(self, name=None):
        """
        Returns a process group with the given name
        If name is None, group is also None indicating the default process group
        If named group does not exist, PhysicsNeMoUndefinedGroupError exception is raised
        """
        if name in self._groups.keys():
            return self._groups[name]
        elif name is None:
            return None
        else:
            raise PhysicsNeMoUndefinedGroupError(name)

    @require_version("torch", "2.4")
    def mesh(self, name=None):
        """
        Return a device_mesh with the given name.
        Does not initialize.  If the mesh is not created
        already, will raise and error

        Parameters
        ----------
        name : str, optional
            Name of desired mesh, by default None
        """

        if name in self._global_mesh.axis_names:
            return self._global_mesh[name]
        elif name is None:
            return self._global_mesh
        else:
            raise PhysicsNeMoUndefinedGroupError(f"Mesh axis {name} not defined")

    def group_size(self, name=None):
        """
        Returns the size of named process group
        """
        if name is None:
            return self._world_size
        group = self.group(name)
        return dist.get_world_size(group=group)

    def group_rank(self, name=None):
        """
        Returns the rank in named process group
        """
        if name is None:
            return self._rank
        group = self.group(name)
        return dist.get_rank(group=group)

    def group_name(self, group=None):
        """
        Returns the name of process group
        """
        if group is None:
            return None
        return self._group_names[group]

    @property
    def broadcast_buffers(self):
        """broadcast_buffers in PyTorch DDP"""
        return self._broadcast_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, broadcast: bool):
        """Setter for broadcast_buffers"""
        self._broadcast_buffers = broadcast

    @property
    def find_unused_parameters(self):
        """find_unused_parameters in PyTorch DDP"""
        return self._find_unused_parameters

    @find_unused_parameters.setter
    def find_unused_parameters(self, find_params: bool):
        """Setter for find_unused_parameters"""
        if find_params:
            warn(
                "Setting `find_unused_parameters` in DDP to true, "
                "use only if necessary."
            )
        self._find_unused_parameters = find_params

    def __str__(self):
        output = (
            f"Initialized process {self.rank} of {self.world_size} using "
            f"method '{self._initialization_method}'. Device set to {str(self.device)}"
        )
        return output

    @classmethod
    def is_initialized(cls) -> bool:
        """If manager singleton has been initialized"""
        return cls._shared_state.get("_is_initialized", False)

    @staticmethod
    def get_available_backend():
        """Get communication backend"""
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            return "nccl"
        else:
            return "gloo"

    @staticmethod
    def initialize_env():
        """Setup method using generic initialization"""
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        if "LOCAL_RANK" in os.environ:
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                local_rank = int(local_rank)
            else:
                local_rank = rank % torch.cuda.device_count()

        else:
            local_rank = rank % torch.cuda.device_count()

        # Read env variables
        addr = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
        )

    @staticmethod
    def initialize_open_mpi(addr, port):
        """Setup method using OpenMPI initialization"""
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="openmpi",
        )

    @staticmethod
    def initialize_slurm(port):
        """Setup method using SLURM initialization"""
        rank = int(os.environ.get("SLURM_PROCID"))
        world_size = int(os.environ.get("SLURM_NPROCS"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="slurm",
        )

    @staticmethod
    def initialize():
        """
        Initialize distributed manager

        Current supported initialization methods are:
            `ENV`: PyTorch environment variable initialization
                https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            `SLURM`: Initialization on SLURM systems.
                Uses `SLURM_PROCID`, `SLURM_NPROCS`, `SLURM_LOCALID` and
                `SLURM_LAUNCH_NODE_IPADDR` environment variables.
            `OPENMPI`: Initialization for OpenMPI launchers.
                Uses `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE` and
                `OMPI_COMM_WORLD_LOCAL_RANK` environment variables.

        Initialization by default is done using the first valid method in the order
        listed above. Initialization method can also be explicitly controlled using the
        `PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD` environment variable and setting it
        to one of the options above.
        """
        if DistributedManager.is_initialized():
            warn("Distributed manager is already intialized")
            return

        addr = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", "12355")
        # https://pytorch.org/docs/master/notes/cuda.html#id5
        # was changed in version 2.2
        if torch.__version__ < (2, 2):
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        else:
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        initialization_method = os.getenv(
            "PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD"
        )
        if initialization_method is None:
            try:
                DistributedManager.initialize_env()
            except TypeError:
                if "SLURM_PROCID" in os.environ:
                    DistributedManager.initialize_slurm(port)
                elif "OMPI_COMM_WORLD_RANK" in os.environ:
                    DistributedManager.initialize_open_mpi(addr, port)
                else:
                    warn(
                        "Could not initialize using ENV, SLURM or OPENMPI methods. Assuming this is a single process job"
                    )
                    DistributedManager._shared_state["_is_initialized"] = True
        elif initialization_method == "ENV":
            DistributedManager.initialize_env()
        elif initialization_method == "SLURM":
            DistributedManager.initialize_slurm(port)
        elif initialization_method == "OPENMPI":
            DistributedManager.initialize_open_mpi(addr, port)
        else:
            raise RuntimeError(
                "Unknown initialization method "
                f"{initialization_method}. "
                "Supported values for "
                "PHYSICSNEMO_DISTRIBUTED_INITIALIZATION_METHOD are "
                "ENV, SLURM and OPENMPI"
            )

        # Set per rank numpy random seed for data sampling
        np.random.seed(seed=DistributedManager().rank)

    @require_version("torch", "2.4")
    def initialize_mesh(
        self, mesh_shape: Tuple[int, ...], mesh_dim_names: Tuple[str, ...]
    ) -> "torch.distributed.DeviceMesh":
        """
        Initialize a global device mesh over the entire distributed job.

        Creates a multi-dimensional mesh of processes that can be used for distributed
        operations. The mesh shape must multiply to equal the total world size, with
        one dimension optionally being flexible (-1).

        Parameters
        ----------
        mesh_shape : Tuple[int, ...]
            Tuple of ints describing the size of each mesh dimension. Product must equal
            world_size. One dimension can be -1 to be automatically calculated.

        mesh_dim_names : Tuple[str, ...]
            Names for each mesh dimension. Must match length of mesh_shape.

        Returns
        -------
        torch.distributed.DeviceMesh
            The initialized device mesh

        Raises
        ------
        RuntimeError
            If mesh dimensions are invalid or don't match world size
        AssertionError
            If distributed environment is not available
        """

        manager = DistributedManager()
        if not manager.distributed:
            raise AssertionError(
                "torch.distributed is unavailable. "
                "Check pytorch build to ensure the distributed package is available. "
                "If building PyTorch from source, set `USE_DISTRIBUTED=1` "
                "to enable the distributed package"
            )

        # Assert basic properties:
        if len(mesh_shape) == 0:
            raise RuntimeError(
                "Device Mesh requires at least one mesh dimension in `mesh_shape`"
            )
        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(
                "mesh_shape and mesh_dim_names must have the same length, but found "
                f"{len(mesh_shape)} and {len(mesh_dim_names)} respectively."
            )
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError("Mesh dimension names must be unique")

        # Check against the total mesh shape vs. world size:
        total_mesh_shape = np.prod(mesh_shape)

        # Allow one shape to be -1
        if -1 in mesh_shape:
            residual_shape = int(self.world_size / (-1 * total_mesh_shape))

            # Replace -1 with the computed size:
            mesh_shape = [residual_shape if m == -1 else m for m in mesh_shape]
            # Recompute total shape:
            total_mesh_shape = np.prod(mesh_shape)

        if total_mesh_shape != self.world_size:
            raise RuntimeError(
                "Device Mesh num elements must equal world size of "
                f"{total_mesh_shape} but was configured by user with "
                f"global size of {self.world_size}."
            )

        # Actually create the mesh:
        self._global_mesh = dist.init_device_mesh(
            "cuda" if self.cuda else "cpu",
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )

        # Finally, upon success, cache the mesh dimensions:
        self._mesh_dims = {key: val for key, val in zip(mesh_dim_names, mesh_shape)}

        return self._global_mesh

    @staticmethod
    def setup(
        rank=0,
        world_size=1,
        local_rank=None,
        addr="localhost",
        port="12355",
        backend="nccl",
        method="env",
    ):
        """Set up PyTorch distributed process group and update manager attributes"""
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        DistributedManager._shared_state["_is_initialized"] = True
        manager = DistributedManager()

        manager._distributed = torch.distributed.is_available()
        if manager._distributed:
            # Update rank and world_size if using distributed
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank

        manager._device = torch.device(
            f"cuda:{manager.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        if manager._distributed:
            # Setup distributed process group
            try:
                dist.init_process_group(
                    backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                    device_id=manager.device,
                )
            except TypeError:
                # device_id only introduced in PyTorch 2.3
                dist.init_process_group(
                    backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                )

        if torch.cuda.is_available():
            # Set device for this process and empty cache to optimize memory usage
            torch.cuda.set_device(manager.device)
            torch.cuda.device(manager.device)
            torch.cuda.empty_cache()

        manager._initialization_method = method

    @staticmethod
    def create_process_subgroup(
        name: str, size: int, group_name: Optional[str] = None, verbose: bool = False
    ):  # pragma: no cover
        """
        Create a process subgroup of a parent process group. This must be a collective
        call by all processes participating in this application.

        Parameters
        ----------
        name : str
            Name of the process subgroup to be created.

        size : int
            Size of the process subgroup to be created. This must be an integer factor of
            the parent group's size.

        group_name : Optional[str]
            Name of the parent process group, optional. If None, the default process group
            will be used. Default None.

        verbose : bool
            Print out ranks of each created process group, default False.

        """
        manager = DistributedManager()
        if not manager.distributed:
            raise AssertionError(
                "torch.distributed is unavailable. "
                "Check pytorch build to ensure the distributed package is available. "
                "If building PyTorch from source, set `USE_DISTRIBUTED=1` "
                "to enable the distributed package"
            )

        if name in manager._groups:
            raise AssertionError(f"Group with name {name} already exists")

        # Get parent group's params
        group = manager._groups[group_name] if group_name else None
        group_size = dist.get_world_size(group=group)
        num_groups = manager.world_size // group_size

        # Get number of sub-groups per parent group
        if group_size % size != 0:
            raise AssertionError(
                f"Cannot divide group size {group_size} evenly into subgroups of"
                f" size {size}"
            )
        num_subgroups = group_size // size

        # Create all the sub-groups
        # Note: all ranks in the job need to create all sub-groups in
        # the same order even if a rank is not part of a sub-group
        manager._group_ranks[name] = []
        for g in range(num_groups):
            for i in range(num_subgroups):
                # Get global ranks that are part of this sub-group
                start = i * size
                end = start + size
                if group_name:
                    ranks = manager._group_ranks[group_name][g][start:end]
                else:
                    ranks = list(range(start, end))
                # Create sub-group and keep track of ranks
                tmp_group = dist.new_group(ranks=ranks)
                manager._group_ranks[name].append(ranks)
                if manager.rank in ranks:
                    # Set group in manager only if this rank is part of the group
                    manager._groups[name] = tmp_group
                    manager._group_names[tmp_group] = name

        if verbose and manager.rank == 0:
            print(f"Process group '{name}':")
            for grp in manager._group_ranks[name]:
                print("    ", grp)

    @staticmethod
    def create_orthogonal_process_group(
        orthogonal_group_name: str, group_name: str, verbose: bool = False
    ):  # pragma: no cover
        """
        Create a process group that is orthogonal to the specified process group.

        Parameters
        ----------
        orthogonal_group_name : str
            Name of the orthogonal process group to be created.

        group_name : str
            Name of the existing process group.

        verbose : bool
            Print out ranks of each created process group, default False.

        """
        manager = DistributedManager()
        if not manager.distributed:
            raise AssertionError(
                "torch.distributed is unavailable. "
                "Check pytorch build to ensure the distributed package is available. "
                "If building PyTorch from source, set `USE_DISTRIBUTED=1` "
                "to enable the distributed package"
            )

        if group_name not in manager._groups:
            raise ValueError(f"Group with name {group_name} does not exist")
        if orthogonal_group_name in manager._groups:
            raise ValueError(f"Group with name {orthogonal_group_name} already exists")

        group_ranks = manager._group_ranks[group_name]
        orthogonal_ranks = [list(i) for i in zip(*group_ranks)]

        for ranks in orthogonal_ranks:
            tmp_group = dist.new_group(ranks=ranks)
            if manager.rank in ranks:
                # Set group in manager only if this rank is part of the group
                manager._groups[orthogonal_group_name] = tmp_group
                manager._group_names[tmp_group] = orthogonal_group_name

        manager._group_ranks[orthogonal_group_name] = orthogonal_ranks

        if verbose and manager.rank == 0:
            print(f"Process group '{orthogonal_group_name}':")
            for grp in manager._group_ranks[orthogonal_group_name]:
                print("    ", grp)

    @staticmethod
    def create_group_from_node(
        node: ProcessGroupNode,
        parent: Optional[str] = None,
        verbose: bool = False,
    ):  # pragma: no cover
        if node.size is None:
            raise AssertionError(
                "Cannot create groups from a ProcessGroupNode that is not fully"
                " populated. Ensure that config.set_leaf_group_sizes is called first"
                " with `update_parent_sizes = True`"
            )

        DistributedManager.create_process_subgroup(
            node.name, node.size, group_name=parent, verbose=verbose
        )
        # Create orthogonal process group
        orthogonal_group = f"__orthogonal_to_{node.name}"
        DistributedManager.create_orthogonal_process_group(
            orthogonal_group, node.name, verbose=verbose
        )
        return orthogonal_group

    @staticmethod
    def create_groups_from_config(
        config: ProcessGroupConfig, verbose: bool = False
    ):  # pragma: no cover

        if torch.__version__ > "2.4":
            warnings.warn(
                "DistributedManager.create_groups_from_config is no longer the most simple "
                "way to organize process groups.  Please switch to DeviceMesh, "
                "and DistributedManager.initialize_mesh",
                category=DeprecationWarning,
                stacklevel=2,
            )

        # Traverse process group tree in breadth first order
        # to create nested process groups
        q = queue.Queue()
        q.put(config.root_id)
        DistributedManager.create_group_from_node(config.root)

        while not q.empty():
            node_id = q.get()
            if verbose:
                print(f"Node ID: {node_id}")

            children = config.tree.children(node_id)
            if verbose:
                print(f"  Children: {children}")

            parent_group = node_id
            for child in children:
                # Create child group and replace parent group by orthogonal group so
                # that each child forms an independent block of processes
                parent_group = DistributedManager.create_group_from_node(
                    child.data,
                    parent=parent_group,
                )

                # Add child ids to the queue
                q.put(child.identifier)

    @atexit.register
    @staticmethod
    def cleanup():
        """Clean up distributed group and singleton"""
        # Destroying group.WORLD is enough for all process groups to get destroyed
        if (
            "_is_initialized" in DistributedManager._shared_state
            and DistributedManager._shared_state["_is_initialized"]
            and "_distributed" in DistributedManager._shared_state
            and DistributedManager._shared_state["_distributed"]
        ):
            if torch.cuda.is_available():
                dist.barrier(device_ids=[DistributedManager().local_rank])
            else:
                dist.barrier()
            dist.destroy_process_group()
        DistributedManager._shared_state = {}
