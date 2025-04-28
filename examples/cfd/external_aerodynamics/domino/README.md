# DoMINO: Decomposable Multi-scale Iterative Neural Operator for External Aerodynamics

DoMINO is a local, multi-scale, point-cloud based model architecture to model large-scale
physics problems such as external aerodynamics. The DoMINO model architecture takes STL
geometries as input and evaluates flow quantities such as pressure and
wall shear stress on the surface of the car as well as velocity fields and pressure
in the volume around it. The DoMINO architecture is designed to be a fast, accurate
and scalable surrogate model for large-scale industrial simulations.

DoMINO uses local geometric information to predict solutions on discrete points. First,
a global geometry encoding is learnt from point clouds using a multi-scale, iterative
approach. The geometry representation takes into account both short- and long-range
depdencies that are typically encountered in elliptic PDEs. Additional information
as signed distance field (SDF), positional encoding are used to enrich the global encoding.
Next, discrete points are randomly sampled, a sub-region is constructed around each point
and the local geometry encoding is extracted in this region from the global encoding.
The local geometry information is learnt using dynamic point convolution kernels.
Finally, a computational stencil is constructed dynamically around each discrete point
by sampling random neighboring points within the same sub-region. The local-geometry
encoding and the computational stencil are aggregrated to predict the solutions on the
discrete points.

A preprint describing additional details about the model architecture can be found here
[paper](https://arxiv.org/abs/2501.13350).

## Dataset

In this example, the DoMINO model is trained using DrivAerML dataset from the
[CAE ML Dataset collection](https://caemldatasets.org/drivaerml/).
This high-fidelity, open-source (CC-BY-SA) public dataset is specifically designed
for automotive aerodynamics research. It comprises 500 parametrically morphed variants
of the widely utilized DrivAer notchback generic vehicle. Mesh generation and scale-resolving
computational fluid dynamics (CFD) simulations were executed using consistent and validated
automatic workflows that represent the industrial state-of-the-art. Geometries and comprehensive
aerodynamic data are published in open-source formats. For more technical details about this
dataset, please refer to their [paper](https://arxiv.org/pdf/2408.11969).

## Training the DoMINO model

To train and test the DoMINO model on AWS dataset, follow these steps:

1. Download the DrivAer ML dataset using the provided `download_aws_dataset.sh` script or
   using the [Hugging Face repo](https://huggingface.co/datasets/neashton/drivaerml).

2. Specify the configuration settings in `conf/config.yaml`.

3. Run `process_data.py`. This will process VTP/VTU files and save them as npy for faster
 processing in DoMINO datapipe. Modify data_processor key in config file. Additionally, run
 `cache_data.py` if using caching to decrease training time. The final processed dataset
 should be divided and saved into 2 directories, for training and validation. Specify these
 directories in `conf/config.yaml`

4. Run `train.py` to start the training. Modify data, train and model keys in config file.
  If using cached data then use `conf/cached.yaml` instead of `conf/config.yaml`.

5. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
  Modify eval key in config file to specify checkpoint, input and output directory.

6. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

## Retraining recipe for DoMINO model

To enable retraining the DoMINO model from a pre-trained checkpoint, follow the steps:

1. Add the pre-trained checkpoints in the resume_dir defined in `conf/config.yaml`.

2. Add the volume and surface scaling factors to the output dir defined in  `conf/config.yaml`.

3. Run `retraining.py` for specified number of epochs to retrain model at a small
 learning rate starting from checkpoint.

4. Run `test.py` to test on `.vtp` / `.vtu`. Predictions are written to the same file.
 Modify eval key in config file to specify checkpoint, input and output directory.

5. Download the validation results (saved in form of point clouds in `.vtp` / `.vtu` format),
   and visualize in Paraview.

## DoMINO model inference on STLs

The DoMINO model can be evaluated directly on unknown STLs using the pre-trained
 checkpoint. Follow the steps outlined below:

1. Run the `inference_on_stl.py` script to perform inference on an STL.

2. Specify the STL paths, velocity inlets, stencil size and model checkpoint
 path in the script.

3. The volume predictions are carried out on points sampled in a bounding box around STL.

4. The surface predictions are carried out on the STL surface. The drag and lift
 accuracy will depend on the resolution of the STL.

## Guidelines for training DoMINO model

1. The DoMINO model allows for training both volume and surface fields using a single model
 but currently the recommendation is to train the volume and surface models separately. This
  can be controlled through the config file.

2. MSE loss for both volume and surface model gives the best results.

3. The surface and volume variable names can change but currently the code only
 supports the variables in that specific order. For example, Pressure, wall-shear
  and turb-visc for surface and velocity, pressure and turb-visc for volume.

4. Bounding box is configurable and will depend on the usecase. The presets are
 suitable for the AWS DriveAer-ML dataset.

5. Integral loss factor is currently set to 0.0 as it adversely impacts the training.

The DoMINO model architecture is used to support the Real Time Wind Tunnel OV Blueprint
demo presented at Supercomputing' 24. Some of the results are shown below.

![Results from DoMINO for RTWT SC demo](../../../../docs/img/domino_result_rtwt.jpg)

## References

1. [DoMINO: A Decomposable Multi-scale Iterative Neural Operator for Modeling Large Scale Engineering Simulations](https://arxiv.org/abs/2501.13350)
