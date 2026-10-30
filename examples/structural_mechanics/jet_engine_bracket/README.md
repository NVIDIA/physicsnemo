# Jet Engine Bracket: Training and Transfer Learning

This example trains a surrogate for the linear-elastic response of jet engine
brackets and adapts it to a new family of designs:

1. **Train** (`train.py`): predict the displacement and the von Mises stress at
   every mesh node of a bracket under four load cases, trained on the 2,138
   generated brackets of DeepJEB. The model is
   `physicsnemo.experimental.models.geotransolver_sdf.GeoTransolverSDF`, a
   [GeoTransolver](https://arxiv.org/abs/2512.20399) in which every node also sees
   the neighbouring nodes and a stencil of signed distances to the surface, which
   tells it the local member thickness. The stock GeoTransolver trains with the
   same recipe for comparison.
2. **Transfer** (`finetune.py`): adapt the DeepJEB model to the human-designed
   SimJEB brackets with low-rank adapters (LoRA, `physicsnemo.experimental.peft`),
   training 0.65 M of its 5.14 M parameters.

## Datasets

| Dataset | Designs | Mesh | Source and license |
| --- | --- | --- | --- |
| DeepJEB | 2,138 generated brackets | 10-node tetrahedra | [Hugging Face](https://huggingface.co/datasets/KAIST-SmartDesignLab/DeepJEB), ODC-By 1.0 |
| SimJEB | 381 human-designed brackets | 4-node tetrahedra | [Harvard Dataverse](https://doi.org/10.7910/DVN/XFUWJG), CC0 1.0 |

Both datasets solve the GE jet engine bracket problem with the same material
(Ti-6Al-4V, E = 113.8 GPa, ν = 0.342) and the same four load cases: vertical,
horizontal and diagonal forces and a torsional moment at the clevis, with the four
bolt holes clamped. Their interfaces differ slightly: every DeepJEB design shares
one interface, 1-3% larger than SimJEB's with the pin 2 mm higher, while the
SimJEB interfaces vary by about 10% between designs. The OptiStruct results give
the displacement and the von Mises stress at every node.

Splits (`splits/`): DeepJEB uses the dataset authors' "latent" split, 1,933
training and 205 test designs, with a seeded 10% of the training designs held out
for validation (1,740 / 193 / 205). SimJEB uses its official 77 test designs and
59 validation designs drawn from the rest (241 / 59 / 77); four designs are
excluded for defects of their mesh or deck, listed with the reason in
`splits/simjeb.json`.

## Setup

```bash
pip install -r requirements.txt

DATA=<data folder>
# 1. Download the parts of both datasets that are used (DeepJEB 77 GB, SimJEB 7.1 GB)
python download_data.py --output $DATA

# 2. Convert them to one DomainMesh per design and load case (bracket_io.py)
python convert_deepjeb.py --input $DATA/deepjeb/extracted \
    --output $DATA/deepjeb/converted --workers 16
python convert_simjeb.py --input $DATA/simjeb/extracted \
    --output $DATA/simjeb/converted --workers 16
```

Each sample holds every mesh node with its region (interior, free surface, clamped
bolt interfaces, loaded clevis interface), the displacement and the von Mises
stress, the boundary triangles, and the load case. The fields are divided by
per-load-case scales of the DeepJEB training designs (`stats/deepjeb_scales.json`,
recomputed by `compute_scales.py`).

## Recipe 1: train on DeepJEB

The model (`conf/model/geotransolver_sdf.yaml`) has 8 GALE blocks of width 192
with 256 physics slices. Every mesh node is a query: it sees its coordinates and
the one-hot of its region, up to 64 neighbouring nodes within 5 mm and 20 mm, and
the signed distance to the surface at a 3x3x3 stencil with spacings of 0.5, 2 and
6 mm. The geometry context is every node voxel-downsampled at 2 mm, so the model
sees where the part is solid, and the global context is the one-hot of the load
case.

```bash
torchrun --nproc_per_node 4 train.py data.root=$DATA/deepjeb/converted
# DeepJEB test, then the model as is on SimJEB test
python evaluate.py data.root=$DATA/deepjeb/converted
python evaluate.py data.root=$DATA/simjeb/converted data.split=splits/simjeb.json \
    data.subset=test output=outputs/deepjeb/evaluation/simjeb_test
```

Training runs 100 epochs (every design under every load case, one sample per GPU
and step) with Muon for the weight matrices and AdamW for the rest, a learning
rate of 2e-3 for four GPUs with a two-epoch warm-up and a cosine decay to 10%,
bfloat16 autocast and activation checkpointing. A rerun resumes from the latest
checkpoint. `model=geotransolver` trains the stock GeoTransolver instead. For a
quick test:

`python train.py data.root=... data.max_designs=8 training.epochs=2`.

## Recipe 2: transfer to SimJEB with LoRA

```bash
python finetune.py data.root=$DATA/simjeb/converted
python evaluate.py adapter=outputs/simjeb_lora/adapter.lora \
    data.root=$DATA/simjeb/converted data.split=splits/simjeb.json \
    output=outputs/simjeb_lora/simjeb_test
```

The pre-trained model is frozen and rank-16 adapters on every linear layer are
trained with AdamW for 30 epochs on the SimJEB training designs, with the
per-sample relative squared error as the loss. `outputs/simjeb_lora/summary.json`
compares the pre-trained and the adapted model on the SimJEB validation and test
designs.

## Hardware, time and memory

| Step | Hardware | Time | Peak GPU memory |
| --- | --- | --- | --- |
| Train `GeoTransolverSDF`, 100 epochs | 4 H100 (80 GB) | about 80 h wall clock | about 75 GB per GPU, on the largest designs |
| Train the stock GeoTransolver, 100 epochs | 4 H100 (80 GB) | about 27 h (16 min per epoch) | |
| Evaluate DeepJEB test (820 samples, float32) | 1 GPU | 8 min | 10 GB |
| Evaluate all SimJEB designs (1,508 samples, float32) | 1 GPU | 5 min | 18 GB (largest design, 642k nodes) |
| LoRA, 30 epochs | 1 GPU (96 GB) | 3.8 h | about 49 GB |

The training times are from runs split into 4-hour cluster jobs; the evaluation
and LoRA times are from one RTX PRO 6000 Blackwell GPU. Activation checkpointing
(on in `conf/model/`) is what makes training fit into 80 GB.

## Results

One run of each recipe. The stock GeoTransolver rows come from this example as
configured; the `GeoTransolverSDF` rows come from a run of the same model and
recipe with the authors' training code, whose checkpoint, converted to
`GeoTransolverSDF`, gives the same errors with this example's `evaluate.py`
(DeepJEB test stress 0.0658 against 0.0659, SimJEB test 0.3827 against 0.3827).
Errors are mean per-sample relative L2 errors over every node, evaluated in float32
(`evaluate.py`); the stress R² is the median over the samples. Results can differ
slightly with the hardware, the number of GPUs and the library version: the LoRA
adaptation, rerun on a different GPU, reproduced the test errors to within 0.005.

| Model | DeepJEB test, displacement / stress | SimJEB test, displacement / stress | SimJEB test stress R² |
| --- | --- | --- | --- |
| `GeoTransolverSDF` | 0.031 / 0.066 | 0.374 / 0.383 | 0.67 |
| + LoRA on SimJEB | | 0.186 / 0.255 | 0.86 |
| Stock GeoTransolver | 0.036 / 0.083 | 0.396 / 0.453 | 0.52 |
| + LoRA on SimJEB | | 0.224 / 0.326 | 0.76 |

On the DeepJEB test designs `GeoTransolverSDF` reaches a median stress R² of 0.99.
On SimJEB, designs unlike any DeepJEB design, the error is four to six times
larger, and largest in members thinner than 3 mm, which DeepJEB hardly contains;
LoRA with the 241 SimJEB training designs brings the test stress error below the
0.295 of the same model trained from scratch on those designs.

## Files

| File | Purpose |
| --- | --- |
| `download_data.py` | download and extract the datasets |
| `nastran.py`, `convert_deepjeb.py`, `convert_simjeb.py`, `bracket_io.py` | read the decks and results, write the samples |
| `compute_scales.py`, `stats/` | per-load-case normalization scales |
| `datapipe.py` | samples as model inputs and normalized targets |
| `train.py`, `evaluate.py`, `finetune.py`, `utils.py` | the recipes |
| `conf/` | Hydra configurations |
| `splits/` | design ids of the splits |

## References

- C. Adams et al., GeoTransolver, [arXiv:2512.20399](https://arxiv.org/abs/2512.20399).
- S. Hong et al., DeepJEB: 3D deep learning-based synthetic jet engine bracket
  dataset, J. Mech. Des. 147(4):041703, 2025.
- E. Whalen et al., SimJEB: Simulated jet engine bracket dataset, Computer
  Graphics Forum 40(5), 2021.
- E. J. Hu et al., LoRA: Low-rank adaptation of large language models, ICLR 2022.
