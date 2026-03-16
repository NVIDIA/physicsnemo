We're working on adding support for physicsnemo Mesh objects to datapipes.  this needs to be supported in a couple ways.

First, we'll support reading mesh objects directly, write off disk, and passing
them instead of tensordicts (since they are, really, a class of tensordicts).
I want to be careful here that we compose transforms properly ... only transforms
meant to consume a mesh should work with meshes.

So, we will need a special organization layer of datapipes to handle mesh
specific objects in the datapipes, I think.  And then we will need some transforms.
I think we will need both generic transforms, as well as a series of augmentations,
which are on-the-fly style transforms that are randomizations.  These should be
separate somehow from decimations.

I'm going to be working with several key datasets as "benchmark" datasets and 
I'll be updating this file as I go with the schema.  So continue to revisit
this file for information.


Here is a view of the DrivaerML file structure:

/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/drivaer_aws/drivaer_data_pnm_mesh/run_1/
├── boundary_1.vtp.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   ├── CpMeanTrim.memmap
│       │   ├── meta.json
│       │   ├── pMeanTrim.memmap
│       │   ├── pPrime2MeanTrim.memmap
│       │   └── wallShearStressMeanTrim.memmap
│       ├── cells.memmap
│       ├── global_data
│       │   ├── meta.json
│       │   └── TimeValue.memmap
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
├── drivaer_1_single_solid.stl.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   └── meta.json
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
├── drivaer_1.stl.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   └── meta.json
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
└── volume_1.vtu.pt
    ├── meta.json
    └── _tensordict
        ├── _cache
        │   ├── cell
        │   │   └── meta.json
        │   ├── meta.json
        │   └── point
        │       └── meta.json
        ├── cell_data
        │   └── meta.json
        ├── global_data
        │   ├── meta.json
        │   └── TimeValue.memmap
        ├── meta.json
        ├── point_data
        │   ├── CpMeanTrim.memmap
        │   ├── CptMeanTrim.memmap
        │   ├── magUMeanNormTrim.memmap
        │   ├── meta.json
        │   ├── microDragMeanTrim.memmap
        │   ├── nutMeanTrim.memmap
        │   ├── pMeanTrim.memmap
        │   ├── pPrime2MeanTrim.memmap
        │   ├── turbulenceProperties:RMeanTrim.memmap
        │   ├── UMeanTrim.memmap
        │   └── UPrime2MeanTrim.memmap
        └── points.memmap

32 directories, 55 files


Here is a view of a SHIFT-SUV "Fastback" file:

⬢ [podman] ❯ tree  $GROUP_LUSTRE/datasets/shift_suv_pnm_mesh/run_00002_fastback/
/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/shift_suv_pnm_mesh/run_00002_fastback/
├── merged_surfaces_filled.stl.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   └── meta.json
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
├── merged_surfaces.stl.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   └── meta.json
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
├── merged_surfaces.vtp.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   ├── meta.json
│       │   ├── Normals.memmap
│       │   ├── pressure_average.memmap
│       │   ├── pressure.memmap
│       │   ├── vtkOriginalCellIds.memmap
│       │   ├── wall_shear_stress_average.memmap
│       │   └── wall_shear_stress.memmap
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   ├── meta.json
│       │   ├── Normals.memmap
│       │   └── vtkOriginalPointIds.memmap
│       └── points.memmap
└── merged_volumes.vtu.pt
    ├── meta.json
    └── _tensordict
        ├── _cache
        │   ├── cell
        │   │   └── meta.json
        │   ├── meta.json
        │   └── point
        │       └── meta.json
        ├── cell_data
        │   └── meta.json
        ├── global_data
        │   └── meta.json
        ├── meta.json
        ├── point_data
        │   ├── meta.json
        │   ├── pressure_average.memmap
        │   ├── pressure.memmap
        │   ├── velocity_average.memmap
        │   └── velocity.memmap
        └── points.memmap

32 directories, 51 files

While here is a SHIFT-SUV "Estate" file:

⬢ [podman] ❯ tree  $GROUP_LUSTRE/datasets/shift_suv_pnm_mesh/run_00004_estate/
/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/shift_suv_pnm_mesh/run_00004_estate/
├── merged_surfaces_filled.stl.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   └── meta.json
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
├── merged_surfaces.stl.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   └── meta.json
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   └── meta.json
│       └── points.memmap
├── merged_surfaces.vtp.pt
│   ├── meta.json
│   └── _tensordict
│       ├── _cache
│       │   ├── cell
│       │   │   └── meta.json
│       │   ├── meta.json
│       │   └── point
│       │       └── meta.json
│       ├── cell_data
│       │   ├── meta.json
│       │   ├── Normals.memmap
│       │   ├── pressure_average.memmap
│       │   ├── pressure.memmap
│       │   ├── vtkOriginalCellIds.memmap
│       │   ├── wall_shear_stress_average.memmap
│       │   └── wall_shear_stress.memmap
│       ├── cells.memmap
│       ├── global_data
│       │   └── meta.json
│       ├── meta.json
│       ├── point_data
│       │   ├── meta.json
│       │   ├── Normals.memmap
│       │   └── vtkOriginalPointIds.memmap
│       └── points.memmap
└── merged_volumes.vtu.pt
    ├── meta.json
    └── _tensordict
        ├── _cache
        │   ├── cell
        │   │   └── meta.json
        │   ├── meta.json
        │   └── point
        │       └── meta.json
        ├── cell_data
        │   └── meta.json
        ├── global_data
        │   └── meta.json
        ├── meta.json
        ├── point_data
        │   ├── meta.json
        │   ├── pressure_average.memmap
        │   ├── pressure.memmap
        │   ├── velocity_average.memmap
        │   └── velocity.memmap
        └── points.memmap

32 directories, 51 files