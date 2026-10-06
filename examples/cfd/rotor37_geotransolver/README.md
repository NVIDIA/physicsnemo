# GeoTransolver for Rotor37 Compressor Blades

This example trains PhysicsNeMo GeoTransolver to predict the surface density,
pressure and temperature of a transonic compressor blade, together with its mass
flow, compression ratio and efficiency, from the blade shape and two operating
conditions. A trained model predicts a complete blade surface, including its
shock, in about 5 ms.

![Reference and predicted pressure jumps](../../../docs/img/rotor37/pressure_jumps.png)

Validation case 613 seen from both sides of the blade. The columns show the
simulated pressure jumps across mesh edges, the predicted jumps and their error.

The example doubles as a walkthrough. Besides the commands to run it, it explains
what makes the problem hard, which approaches come to mind first, why they
struggle here, and how each part of the recipe answers one difficulty. The same
reasoning applies to many surrogate models of engineering surfaces.

## The recipe at a glance

```text
blade surface + operating conditions
  -> 66 case features and 2,048 surface points       (step 4)
  -> GeoTransolver                                   (step 5)
  -> 3 compressor outputs and 349 mode coefficients  (step 1)
  -> fixed decoder, exp and ideal-gas relation       (step 2)
  -> density, pressure and temperature on all 29,773 vertices
```

| Difficulty | How the recipe handles it | Step |
| --- | --- | --- |
| 800 training cases but 89,319 field values per case | Predict coefficients of fixed spatial modes | 1 |
| Pressure, temperature and density must be positive and consistent | Log fields, exponentiation and density from the gas law | 2 |
| Pointwise errors hardly notice a smeared shock | Edge-aware modes and a pressure jump loss | 3 |
| The blade shape is a 3D surface | Principal components of coordinates and normals, plus surface points | 4 |
| Choosing the network | GeoTransolver, with a multilayer perceptron baseline | 5 |
| Little data for a large network | Zero-initialized inputs and outputs, balanced loss | 6 |

## Problem overview

NASA Rotor 37 is a transonic axial compressor rotor and a standard test case for
turbomachinery simulation. Near the blade tip the relative flow is supersonic,
so a shock forms on the blade where the surface pressure rises sharply over a
few mesh cells. The position and strength of the shock depend on the blade shape
and the operating point, and they strongly influence losses and efficiency.

Each Reynolds-averaged Navier-Stokes simulation is expensive, while design
studies need many blade shapes and operating points. A surrogate that maps a
blade and its operating conditions directly to the surface fields and the
compressor performance makes that exploration interactive.

| Inputs | Outputs |
| --- | --- |
| Blade surface coordinates and normals | Surface density, pressure and temperature |
| Rotational speed `Omega` and pressure parameter `P` | Mass flow, compression ratio and efficiency |

## Dataset

The [PLAID Rotor37 dataset](https://huggingface.co/datasets/PLAID-datasets/Rotor37)
contains simulations of 1,200 blade geometries and operating points. All blade
surfaces share one mesh with 29,773 vertices, 29,664 quadrilateral faces and
59,436 unique edges, so a vertex index refers to the same location on every
blade. The source does not declare physical units, so errors are reported in the
original numerical scales.

The 1,000 labeled cases are split into 800 training, 100 validation and 100 test
cases with split seed 42. The 200 official test cases have withheld targets.
Every normalization, geometry encoding and field basis is fitted to the 800
training cases only, so no information from evaluation cases reaches the model.

The dataset is owned by Safran and distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## What makes this problem hard

1. **Few examples, very large outputs.** Each of the 800 training simulations
   has 89,319 field values. A model with a free output per vertex has far more
   freedom than the data can pin down.
2. **Sharp features in smooth fields.** The fields are smooth almost everywhere
   but jump across the shock. Pointwise metrics barely notice a smeared shock,
   yet the shock is what an engineer looks at first.
3. **Physical constraints.** Pressure, temperature and density must be positive
   and satisfy the ideal-gas relation. Predicting them independently can break
   both.
4. **Geometry as an input.** The blade shape must reach the network in a form
   compact enough to learn from 800 examples yet detailed enough to move the
   shock.
5. **Integrated quantities.** Three compressor outputs per case must be learned
   alongside tens of thousands of field values.

One property makes the problem easier. **All blades share the same mesh**, so
fields of different cases can be compared vertex by vertex. Every step below
builds on this.

## The approach, step by step

### Step 1. What should the network predict?

**Coefficients of fixed spatial modes, not values at every vertex.**

The obvious alternative is to predict density, pressure and temperature at each
vertex, as point cloud and graph models do. That is the right tool when every
case has its own mesh. Here it spends capacity relearning that neighboring
vertices behave alike, and nothing ties the 29,773 predictions of a blade into a
coherent field.

Because the meshes correspond, the training fields can be stacked and
decomposed by proper orthogonal decomposition into a mean field and spatial
modes ordered by how much variation they explain. 248 pressure modes and 101
temperature modes keep 99.9% of the training variation. The network predicts
these 349 coefficients, and a fixed decoder without learned parameters turns
them into complete fields.

Every prediction is now a combination of patterns seen in real simulations,
which keeps fields coherent and shrinks the learning problem. The price is that
the model only applies to blades meshed with the same vertex ordering, and it
cannot produce patterns absent from the training fields.

### Step 2. How do predictions stay physical?

**Predict log fields and compute density from the gas law.**

The modes describe log pressure and log temperature, and the decoder
exponentiates them. Pressure and temperature are therefore always positive, and
the multiplicative changes between operating points become additive, which suits
a linear combination of modes.

Density is never predicted. It follows from the ideal-gas relation

$$
\hat\rho = \frac{\hat p}{R\hat T},
$$

with the gas constant $R$ fitted to the training fields, which satisfy the
relation to a relative deviation below $2\times10^{-7}$. A third independent
output could contradict the other two, while this construction holds exactly.
Density is still compared with its reference during training, so its errors
also correct pressure and temperature.

### Step 3. How is the shock kept sharp?

**Measure pressure through its differences across mesh edges.**

A shock is a large pressure difference between neighboring vertices. A slightly
blurred or shifted shock increases the pointwise error only a little, so a model
trained on pointwise errors alone has little reason to get it right. The recipe
uses edge differences in two places.

- **In the basis.** The pressure modes are fitted in a norm that weighs edge
  differences equally with pointwise values. Modes describing sharp transitions
  carry little pointwise energy, and this norm keeps them.
- **In the loss.** A jump term compares predicted and simulated pressure
  differences over the set $E$ of mesh edges and penalizes a shock with the
  wrong strength or position.

$$
\mathcal L_{\mathrm{jump}} =
\frac{1}{C_p |E|}\sum_{(i,j)\in E}
\left[\frac{(\hat p_i-\hat p_j)-(p_i-p_j)}{\sigma_p}\right]^2
$$

Here $\sigma_p$ is the training pressure standard deviation and $C_p$ is the
value of the same mean for a prediction equal to the training mean pressure,
which puts the term on the scale of the others. Differences cannot see a
constant pressure offset, so the pointwise field loss stays to fix the overall
level.

### Step 4. How does the geometry enter?

**As 66 compact case features plus a cloud of surface points.**

All 29,773 vertex positions as a flat vector would give the network far more
inputs than training cases. The blade shapes vary in a structured way, so
principal component analysis again gives a compact description.

| Features | Count | Why |
| --- | --- | --- |
| Operating conditions `Omega` and `P` | 2 | They set the flow regime |
| Principal components of the displacement from the mean blade | 32 | Overall blade shape |
| Principal components of the surface normals | 32 | Curvature such as the leading edge, which barely changes coordinates |

Each feature is standardized with its training mean and standard deviation.
Without the normal features, the error on the strongest pressure jumps grows by
about half. The number of components is a trade-off. Too few lose shape detail
that moves the shock, while too many add weak components that are mostly noise
yet receive full weight after standardization. Halving or quadrupling the count
both make validation errors clearly worse.

The network also receives 2,048 surface vertices with their coordinates,
normals and displacements, which GeoTransolver reads as a geometry context.

### Step 5. Which network?

**GeoTransolver, with a multilayer perceptron as baseline.**

After steps 1 to 4 the task is to map 66 case features, plus the surface points,
to 352 numbers. Three families of PhysicsNeMo models fit.

- **Multilayer perceptron.** Reads only the 66 features. The simplest choice and
  the natural baseline, included as `model=mlp`.
- **Fourier neural operator.** Treats the blade surface as an image. Apart from
  the tip, the Rotor37 mesh is a structured sheet of 133 by 216 vertices wrapped
  around the blade, so spectral convolutions can run on the surface directly.
- **GeoTransolver.** Uses the 66 features as its single query token and its
  global context, and reads the surface points through cross-attention. With
  four GALE layers, 512 hidden channels, four attention heads and 32 slices it
  has 8,764,988 parameters. Its 352 outputs are the three compressor outputs and
  the 349 mode coefficients.

GeoTransolver is the most accurate. A multilayer perceptron with the same number
of parameters, trained with the same modes, losses and schedule, has 70 to 80%
higher field errors and strongest-jump error, and even a GeoTransolver shrunk to
3.5 million parameters beats it on every metric. A Fourier neural operator on
the surface grid matches GeoTransolver's overall jump error only with about four
times as many parameters, and stays less accurate on the strongest jumps and the
fields.

The principal components already carry most of the shape information. Replacing
each blade's surface points with the training mean blade changes GeoTransolver's
errors by only a few percent. The surface points matter more when shape
variations are too rich for a few dozen components.

### Step 6. How to train with only 800 examples?

**Start from the mean prediction and balance the loss.**

*Initialization.* With so few cases, how the network starts matters more than
the usual regularizers. The weights that read the 66 case features and the
output layer start at zero, so the untrained model predicts the training mean
for every blade and its sensitivity to each input grows only as far as the data
support it. Standard random initialization makes the network respond from the
start to every input direction, including weak principal components, and that
sensitivity persists. Starting at zero lowers validation errors by 15 to 37%
across fields, shock and compressor outputs. Dropout and input noise, by
contrast, make the model worse.

*Loss.* The total loss is

$$
\mathcal L = \mathcal L_{\mathrm{fields}} + 4\,\mathcal L_{\mathrm{globals}} +
\tfrac12\mathcal L_{\mathrm{coeff},p} + \tfrac12\mathcal L_{\mathrm{coeff},T} +
\mathcal L_{\mathrm{jump}}
$$

| Term | Compares | Why |
| --- | --- | --- |
| $\mathcal L_{\mathrm{fields}}$ | Decoded density, pressure and temperature, normalized | The quantity users evaluate |
| $\mathcal L_{\mathrm{globals}}$ | The three normalized compressor outputs | Weight 4 keeps three numbers from being drowned out by the fields |
| $\mathcal L_{\mathrm{coeff}}$ | Predicted and training mode coefficients | A direct target per mode, weighted by mode energy for pressure and its square root for temperature |
| $\mathcal L_{\mathrm{jump}}$ | Pressure differences across mesh edges | Sharp, correctly placed shocks, step 3 |

*Optimization.* Training runs 300 epochs at batch size 16, which gives 15,000
updates, with AdamW, an initial learning rate of 0.001 decayed to 0.00001 by a
cosine schedule, weight decay 0.0001 and gradient clipping at 1. Larger learning
rates speed up early progress, but with the exponential decoder they can make
training diverge midway.

## Prerequisites

Install PhysicsNeMo and the additional dependencies of this example.

```bash
pip install -r requirements.txt
```

## Getting Started

Run all commands from this directory.

### Data preparation

Running this code will automatically download data from
[https://huggingface.co/datasets/PLAID-datasets/Rotor37](https://huggingface.co/datasets/PLAID-datasets/Rotor37).
Before you run the code, please confirm the content of the dataset and licensing
is appropriate for your intended use.

```bash
python prepare_data.py --data-dir data/rotor37
```

This downloads a pinned revision of the dataset into `data/rotor37/raw`,
decodes every case, splits the labeled cases and fits the normalization, the
geometry principal components and the field modes on the training cases. The
results go to `data/rotor37/processed`. Download and preparation need about
10 GB of disk space.

### Training

```bash
python train.py
```

Training takes about six minutes on one NVIDIA L40 GPU and writes the
configuration, the training history and checkpoints to `runs/geotransolver`.
`python train.py model=mlp` trains the baseline into `runs/mlp`. Any setting can
be changed on the command line, for example `seed=43` or
`training.num_epochs=600`.

On several GPUs, set the per-process batch size to 16 divided by the number of
GPUs so that the global batch size stays at 16.

```bash
torchrun --standalone --nproc_per_node=<NUM_GPUS> train.py training.batch_size=<BATCH_SIZE>
```

### Evaluation

```bash
python evaluate.py 'evaluation.export_samples=[613]'
```

Evaluation runs the final checkpoint in full float32 precision on the validation
cases and writes `metrics.json`, per-case errors in `cases.csv` and the exported
predictions to `runs/geotransolver/evaluation/validation`. Add `model=mlp` for
the baseline, `evaluation.split=test` for the test cases, or
`evaluation.split=official_test` to predict the 200 official test cases.

### Visualization

```bash
python visualize.py
```

This writes the compressor output parity plot and the pressure jumps of case 613
to `outputs/figures`.

## Results

Both models were trained with the default configuration and seed 42 and
evaluated on the 100 validation cases. Relative L2 errors are computed per case
and averaged. Jump errors are root mean square errors of the edge pressure
differences divided by $\sigma_p$, over all 59,436 edges and over the 1% of
edges with the largest simulated differences in each case, which concentrate at
the shock and the leading edge.

| Metric | GeoTransolver | MLP baseline |
| --- | --- | --- |
| Density relative L2 | 0.448% | 0.803% |
| Pressure relative L2 | 0.469% | 0.838% |
| Temperature relative L2 | 0.076% | 0.105% |
| Pressure jump RMSE / $\sigma_p$, all edges | 0.0055 | 0.0077 |
| Pressure jump RMSE / $\sigma_p$, strongest 1% | 0.0102 | 0.0173 |
| Mass flow RMSE | 0.0206 | 0.0322 |
| Compression ratio RMSE | 0.0021 | 0.0038 |
| Efficiency RMSE | 0.00079 | 0.00100 |
| Parameters | 8,764,988 | 9,009,248 |

![Predicted and true compressor outputs](../../../docs/img/rotor37/compressor_parity.png)

The parity plot compares GeoTransolver's predicted and true compressor outputs
for all 100 validation cases. In the figure at the top of this page the
predicted shock sits at the simulated position with the simulated strength, and
the remaining error concentrates along the shock and the leading edge.

The metrics of both models are in `reference_results/geotransolver` and
`reference_results/mlp`.

## Limitations and next steps

- The field modes tie the model to the Rotor37 mesh. Blades with other meshes
  must first be mapped to a common mesh, for example by mesh morphing as in the
  reference below.
- Training errors are several times lower than validation errors, so more
  simulations are the most direct way to improve accuracy.
- The compressor outputs are smooth functions of the 66 case features. A
  Gaussian process on these features predicts them more accurately than the
  network, while the network predicts the fields more accurately. Combining the
  two is a natural extension.

## Tests

```bash
pytest tests
```

The tests check source decoding, the training-only preparation, the decoder and
the jump loss, and train and evaluate small GeoTransolver and multilayer
perceptron models on synthetic blades.

## References

- [PLAID Rotor37 dataset](https://huggingface.co/datasets/PLAID-datasets/Rotor37)
- [MMGP, a mesh morphing Gaussian process method evaluated on Rotor37](https://arxiv.org/abs/2305.12871)
