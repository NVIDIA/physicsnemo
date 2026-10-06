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

**A short list of numbers that describes the whole field, instead of a value at
every vertex.**

*The direct approach.* Ask the network for the pressure at one vertex, and
repeat for all 29,773 vertices. Point cloud and graph networks work this way,
and they are the right choice when every case has its own mesh. Here two things
go wrong. With only 800 examples the network must learn on its own that
neighboring vertices have similar values, and nothing forces its 29,773 answers
to form one smooth, consistent field.

*The shortcut this dataset allows.* Every blade uses the same mesh, so a
pressure field is simply a list of 29,773 numbers in a fixed order, and the 800
training fields can be compared entry by entry. Seen this way, the fields do not
vary arbitrarily. They change in a limited number of typical ways, such as the
overall pressure level rising with the operating point or the shock shifting
along the blade.

Proper orthogonal decomposition, the same method as principal component
analysis, finds these typical patterns automatically. It returns a **mean field**,
the average over the training blades, and a set of **modes**, fixed spatial
patterns ordered from most to least important. Each field is then approximated
by the mean plus a weighted sum of modes,

$$
\log p(\mathbf x) \approx \overline{\log p}(\mathbf x) +
\sum_{k=1}^{248} c_k\, \phi_k(\mathbf x).
$$

The mean and the modes $\phi_k$ are computed once from the training data and
never change. Only the weights $c_k$, called **coefficients**, differ from blade
to blade. 248 pressure modes and 101 temperature modes reproduce 99.9% of the
variation in the training fields. Step 2 explains why the sum describes the
logarithm of the field.

The network therefore predicts 349 coefficients per blade, plus the three
compressor outputs, and a fixed decoder evaluates the sum above on all 29,773
vertices.

| | Value at every vertex | Mode coefficients |
| --- | --- | --- |
| Numbers predicted per blade | 89,319 | 349 |
| Smooth, consistent fields | Must be learned from the data | Built in, since every mode is a pattern from real simulations |
| Blades on a different mesh | Supported | Not supported without first mapping them to this mesh |
| Patterns never seen in training | Possible | Only combinations of training patterns |

The last two rows are the price of this choice. For design studies on one fixed
mesh, as here, it is a good trade.

### Step 2. How do predictions stay physical?

**Build positivity and the gas law into the decoder, so the network cannot break
them.**

*The problem.* A network that outputs pressure, temperature and density directly
can produce a negative pressure for an unusual blade, and its three outputs can
disagree with the ideal-gas relation that the simulations satisfy exactly.
Penalizing such errors in the loss makes them rare but never impossible.

*Positivity.* The modes of step 1 describe the logarithm of pressure and of
temperature, and the decoder takes the exponential of the result. Whatever
coefficients the network predicts, the decoded pressure and temperature are
positive. The logarithm also suits the physics. A change of operating point
scales the pressure, and a scaling becomes a simple shift in the logarithm, which
a sum of modes represents easily.

*Consistency.* Density is never predicted. The decoder computes it from the
ideal-gas relation

$$
\hat\rho = \frac{\hat p}{R\hat T},
$$

where the gas constant $R$ is fitted to the training fields, which follow this
relation to within a relative deviation of $2\times10^{-7}$. The three fields
therefore agree exactly in every prediction. Density is still compared with the
simulation during training, and because it depends on both pressure and
temperature, its errors also improve those two fields.

### Step 3. How is the shock kept sharp?

**Compare pressure differences between neighboring vertices, not only pressure
values.**

*The problem.* A shock is a jump in pressure between neighboring vertices.
Imagine a predicted shock that is correct everywhere except that it is spread
over five cells instead of one, or shifted by one cell. Only the few vertices at
the shock are wrong, so the average pointwise error barely changes, although
the prediction now misses the most important feature of the flow. A model
trained on pointwise errors alone has little reason to get shocks right.

*The idea.* The pressure difference across each mesh edge is large exactly at
the shock and small elsewhere. Measuring errors in these differences makes a
blurred or misplaced shock expensive. The recipe uses them in two places.

- **When choosing the modes.** Proper orthogonal decomposition keeps the
  patterns that explain the most variation. A sharp shock occupies few vertices,
  so measured pointwise it explains little and its patterns would be dropped.
  The pressure modes are therefore selected by a measure that counts the edge
  differences as much as the pointwise values, which keeps the patterns that
  describe sharp transitions.
- **In the loss.** A jump term compares the predicted and simulated pressure
  differences over all mesh edges $E$,

$$
\mathcal L_{\mathrm{jump}} =
\frac{1}{C_p |E|}\sum_{(i,j)\in E}
\left[\frac{(\hat p_i-\hat p_j)-(p_i-p_j)}{\sigma_p}\right]^2 .
$$

Pressure is divided by its training standard deviation $\sigma_p$, and the sum
is divided by $C_p$, its value when every blade is predicted with the training
mean pressure. A value of 1 therefore means no better than the mean, which keeps
the term on the same scale as the other losses.

Differences cannot detect an error that shifts the whole pressure field by a
constant, so the ordinary pointwise loss stays in place to fix the pressure
level.

### Step 4. How does the geometry enter?

**As 66 numbers that summarize each blade, plus a sample of surface points.**

*The problem.* The blade shape is the main design variable, but its raw form is
29,773 points in 3D. Feeding all of them to the network as one long vector gives
it about 90,000 inputs to relate to only 800 examples.

*The idea.* The training blades are variations of one design, so their shapes
differ in a limited number of typical ways, just like the fields in step 1.
Principal component analysis finds these ways of deforming the blade, and each
blade is then described by how much of each deformation it contains.

| Features | Count | What they describe |
| --- | --- | --- |
| Operating conditions `Omega` and `P` | 2 | The flow regime |
| Components of the displacement from the average blade | 32 | The overall shape |
| Components of the surface normals | 32 | Local curvature, such as the shape of the leading edge |

Normals are included because small, sharp shape details barely move the surface
coordinates yet change the surface direction strongly, and they influence where
the shock forms. Without the normal features, the error on the strongest
pressure jumps grows by about half. Every feature is standardized, meaning
shifted and scaled to zero mean and unit spread over the training blades, so
that all inputs reach the network on the same scale.

*How many components?* Too few lose shape details that move the shock. Too many
add components that capture tiny, mostly random variations, and standardization
gives them the same weight as the important ones. Halving or quadrupling the 32
components per group both make validation errors clearly worse.

In addition to the 66 features, the network receives 2,048 points of the blade
surface with their coordinates, normals and displacements.

### Step 5. Which network?

**GeoTransolver, with a multilayer perceptron as baseline.**

After steps 1 to 4 the task is to map the 66 features and the surface points of
a blade to 352 numbers, three compressor outputs and 349 mode coefficients.
Three families of PhysicsNeMo models suit this task.

- **Multilayer perceptron.** A stack of fully connected layers that reads only
  the 66 features. It is the simplest choice and the natural baseline, included
  as `model=mlp`.
- **Fourier neural operator.** Learns convolutions in Fourier space over a grid,
  treating the blade surface as an image. Apart from the tip, the Rotor37 mesh
  is a regular sheet of 133 by 216 vertices wrapped around the blade, so this
  is possible here.
- **GeoTransolver.** A transformer for physics on geometry. It builds an internal
  representation of the case from the 66 features and refines it by attending
  to the surface points, which lets it look up local shape information wherever
  it helps. The configuration used here has four GALE layers, 512 hidden
  channels, four attention heads and 32 slices.

All three were trained with the same modes, losses and schedule. Averages over
three training seeds on the validation cases are

| Network | Parameters | Pressure relative L2 | Jump RMSE / $\sigma_p$ | Strongest 1% jump RMSE / $\sigma_p$ |
| --- | --- | --- | --- | --- |
| GeoTransolver | 8.8 million | 0.457% | 0.0055 | 0.0107 |
| GeoTransolver, smaller | 3.5 million | 0.484% | 0.0056 | 0.0110 |
| Fourier neural operator | 33.6 million | 0.495% | 0.0055 | 0.0119 |
| Multilayer perceptron | 9.0 million | 0.853% | 0.0078 | 0.0176 |

The multilayer perceptron, at the same size as GeoTransolver, has about 85%
higher field error and 65% higher error on the strongest jumps. The Fourier
neural operator matches GeoTransolver's overall jump error only with four times
as many parameters, and is less accurate at the shock. Even the smaller
GeoTransolver beats both, so the advantage comes from the architecture and not
from its size.

How much do the surface points contribute? Replacing each blade's points with
those of the average blade changes GeoTransolver's errors by only a few percent,
because the 66 features already carry most of the shape information. The
surface points become more important when the shapes vary too richly for a few
dozen components.

### Step 6. How to train with only 800 examples?

**Start the network from the average answer, balance the loss terms and keep the
learning rate moderate.**

*Initialization.* A freshly initialized network has random weights, so it
responds to every input from the start, including the weakest geometry
components, which mostly carry noise. With 800 examples that early sensitivity
is never fully unlearned. Here the weights that read the 66 features and the
weights of the output layer start at zero instead. The untrained model then
predicts the average training blade for every input, and its sensitivity to
each feature grows only as far as the training data support it. This lowers
validation errors by 15 to 37% across fields, shock and compressor outputs.
Common regularizers such as dropout and input noise, by contrast, make this
model worse.

*Loss.* The training loss adds up four kinds of error

$$
\mathcal L = \mathcal L_{\mathrm{fields}} + 4\,\mathcal L_{\mathrm{globals}} +
\tfrac12\mathcal L_{\mathrm{coeff},p} + \tfrac12\mathcal L_{\mathrm{coeff},T} +
\mathcal L_{\mathrm{jump}} .
$$

| Term | What it compares | Why it is there |
| --- | --- | --- |
| $\mathcal L_{\mathrm{fields}}$ | Decoded density, pressure and temperature at every vertex | These fields are what users evaluate |
| $\mathcal L_{\mathrm{globals}}$ | Mass flow, compression ratio and efficiency | Only three numbers per blade, so weight 4 keeps them from being drowned out by the fields |
| $\mathcal L_{\mathrm{coeff},p}$, $\mathcal L_{\mathrm{coeff},T}$ | Predicted and true mode coefficients | A direct target for every mode, with important modes weighted more |
| $\mathcal L_{\mathrm{jump}}$ | Pressure differences across mesh edges | Keeps shocks sharp and in place, see step 3 |

All quantities are normalized by their training statistics so that the terms
are comparable.

*Optimization.* Training runs 300 epochs at batch size 16, which gives 15,000
parameter updates. It uses the AdamW optimizer with a learning rate that starts
at 0.001 and decays to 0.00001 along a cosine curve, weight decay 0.0001 and
gradient clipping at 1. Larger learning rates speed up early progress, but
because the decoder exponentiates its input, a large step can blow up the
predicted fields and make training diverge midway.

## Prerequisites

Besides PhysicsNeMo, the example needs huggingface-hub to download the dataset,
pyarrow to read its Parquet files and matplotlib for the figures.

```bash
pip install -r requirements.txt
```

## Getting Started

Run all commands from this directory.

### Data preparation

The dataset is available at
[https://huggingface.co/datasets/PLAID-datasets/Rotor37](https://huggingface.co/datasets/PLAID-datasets/Rotor37).

Running this code will automatically download data from
[https://huggingface.co/datasets/PLAID-datasets/Rotor37](https://huggingface.co/datasets/PLAID-datasets/Rotor37).
Before you run the code, please confirm the content of the dataset and licensing
is appropriate for your intended use.

```bash
python download_data.py
python prepare_data.py
```

`download_data.py` downloads revision `bac06c0caa7254120eecc6711a5fb85c58dfbdbc`
of the dataset, about 4 GB, into `data/rotor37/raw`. To use a copy downloaded by
other means, place it there instead, so that the dataset card is at
`data/rotor37/raw/README.md` and the samples are at
`data/rotor37/raw/data/*.parquet`.

`prepare_data.py` reads these files without network access. It decodes every
case, splits the labeled cases and fits the normalization, the geometry
principal components and the field modes on the training cases. The results go
to `data/rotor37/processed`, which needs about 1.5 GB more.

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
