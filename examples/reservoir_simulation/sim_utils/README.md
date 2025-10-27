# Simulation Utilities

## Overview

The `sim_utils` package provides utilities for processing ECL/IX style binary
output files to prepare datasets for training. These scripts can read industry
standard simulator output formats (ECLIPSE, IX, OPM) and convert them into
various data structures suitable for different ML architectures.

## Supported Formats

- `.INIT`
- `.EGRID`
- `.UNRST` or `.X00xx`
- `.UNSMRY` or `.S00xx`

## Modules

### `ecl_reader.py`

Main class for reading ECLIPSE-style binary output files.

**Usage**:

```python
from sim_utils import EclReader

# Initialize reader with case name
reader = EclReader("path/to/CASE.DATA")

# Read static properties
init_data = reader.read_init(["PORV", "PERMX", "PERMY", "PERMZ"])

# Read grid geometry
egrid_data = reader.read_egrid(["COORD", "ZCORN", "FILEHEAD", "NNC1", "NNC2"])

# Read dynamic properties (all timesteps)
restart_data = reader.read_restart(["PRESSURE", "SWAT", "SGAS"])
```

**Common Keywords**:

Static properties (INIT):

- `PORV`: Pore volume
- `PERMX`, `PERMY`, `PERMZ`: Permeability in X, Y, Z directions
- `PORO`: Porosity
- `TRANX`, `TRANY`, `TRANZ`: Transmissibility in X, Y, Z directions

Dynamic properties (UNRST):

- `PRESSURE`: Cell pressure
- `SWAT`: Water saturation
- `SGAS`: Gas saturation
- `SOIL`: Oil saturation

Grid geometry (EGRID):

- `COORD`: Grid pillar coordinates
- `ZCORN`: Grid corner depths
- `FILEHEAD`: File header information
- `NNC1`, `NNC2`: Non-neighboring connections

### `grid.py`

**Grid** - Handles reservoir grid structure and operations.

**Features**:

- Grid dimensions and active cells
- Cell center coordinates computation
- Connection/edge computation for graph construction
- Aggregating directional transmissibilities for edge features
- Non-Neighboring Connections (NNC)
- Well completion arrays

**Usage**:

```python
from sim_utils import Grid

# Initialize grid from simulation data
grid = Grid(init_data, egrid_data)

# Get connections and transmissibilities for graph construction
connections, transmissibilities = grid.get_conx_tran()

# Create completion arrays for wells
completion_inj, completion_prd = grid.create_completion_array(wells)

# Access grid properties
print(f"Grid dimensions: {grid.nx} x {grid.ny} x {grid.nz}")
print(f"Active cells: {grid.nact}")
print(f"Cell coordinates: X={grid.X}, Y={grid.Y}, Z={grid.Z}")
```

### `well.py`

**Well** and **Completion** - Well and completion data structures. Typically,
use results from `UNRST` (including well name, type, status, I, J, K) to
instantiate the object.

**Usage**:

```python
from sim_utils import Well, Completion

# Create a well
well = Well(name="INJ1", type_id=3, stat=1)  # Water injector

# Add completions
well.add_completion(
    I=10,          # Grid I-index
    J=10,          # Grid J-index
    K=5,           # Grid K-index
    dir=3,         # Direction (1=X, 2=Y, 3=Z)
    stat=1,        # Status (1=OPEN)
    conx_factor=1.0  # Connection factor
)

# Check well properties
print(f"Well type: {well.type}")  # 'INJ' or 'PRD'
print(f"Well status: {well.status}")  # 'OPEN' or 'SHUT'
print(f"Number of completions: {len(well.completions)}")
```
