PhysicsNeMo Deploy
===================

.. automodule:: physicsnemo.deploy
.. currentmodule:: physicsnemo.deploy

Deploying models trained in PhysicsNeMo
--------------------------------------

Application developers can deploy PhysicsNeMo either as the training framework or 
deploy inference recipes using models trained in PhysicsNeMo into their applications.
PhysicsNeMo is written natively in python and you can use the standard python packaging
and deployment practices to productize your applications. It is provided under the
`Apache License 2.0 <https://github.com/NVIDIA/physicsnemo/blob/main/LICENSE.txt>`_.

physicsnemo.deploy.onnx
-----------------------

ONNX is a standard format for representing and exchanging machine learning models in 
other frameworks or environments without significant re-work. The
physicsnemo.deploy.onnx module translates a model from physicsnemo.model and converts
it into an ONNX graph.

The exported model can be consumed by any of the many runtimes that support ONNX, including Microsoft’s ONNX Runtime.

Next example shows how to export a simple model.

.. code:: python

    from physicsnemo.deploy.onnx import export_to_onnx_stream, run_onnx_inference
    from physicsnemo.models.mlp import FullyConnected

     model = FullyConnected(
        in_features=32,
        out_features=8,
        num_layers=1,
        layer_size=8,
    )
    x = torch.randn(4, 32).to(device)
    y = model(x) # Get PyTorch output

    onnx_stream = export_to_onnx_stream(model)
    ort_y = run_onnx_inference(onnx_stream, x)
    ort_y = torch.Tensor(ort_y[0])

    # test the output
    assert torch.allclose(y, ort_y, atol=1e-4)

ONNX
----
.. automodule:: physicsnemo.deploy.onnx.utils
    :members:
    :show-inheritance:
