# Fourier Neural Operater for Darcy Flow

This example demonstrates how to set up a data-driven model for a 2D Darcy flow using
the Fourier Neural Operator (FNO) architecture inside of PhysicsNeMo.
Training progress can be tracked through [MLFlow](https://mlflow.org/docs/latest/index.html).
This example runs on a single GPU, go to the
`darcy_nested_fno` example for exploring a multi-GPU training.

## Getting Started

### Prerequisites

Install the required dependencies by running below:

```bash
pip install -r requirements.txt
```

To train the model, run

```bash
python train_fno_darcy.py
```

training data will be generated on the fly.

Progress can be monitored using MLFlow. Open a new terminal and navigate to the training
directory, then run:

```bash
mlflow ui -p 2458
```

View progress in a browser at <http://127.0.0.1:2458>

If training on a remote machine, set up a ssh tunnel to
the server with `LocalForward 8080 your_remote_machine_addr:8080`.
ssh to the server via the specified port, in this case `8080`, navigate to the training
directory and launch mlflow server

```bash
mlflow server --host 0.0.0.0 --port 8080
```

On your local machine, open a browser and connect to `localhost:8080`.

## Additional Information

## References

- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
