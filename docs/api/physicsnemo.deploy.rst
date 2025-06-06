PhysicsNeMo Deploy
===================
Application developers can deploy PhysicsNeMo either as the training framework or deploy inference recipes using models trained in PhysicsNeMo into their applications. PhysicsNeMo is written natively in python and you can use the standard python packaging and deployment practices to productize your applications. It is provided under the [Apache License 2.0](https://github.com/NVIDIA/physicsnemo/blob/main/LICENSE.txt). 

You can also deploy PhysicsNeMo as a container wrapped up in Triton

## Deploying modess trained in PhysicsNeMo 
For scenarios where you are interested in only deploying the inference recipes, you would want to minimize the footprint by exporting the trained models. Open Neural Network eXchange (ONNX) is an open standardized format for representing and exchanging machine learning models in other frameworks or environments without significant re-work.  The physicsnemo.deploy.onnx module translates a model from physicsnemo.model and converts it into an ONNX graph.

The exported model can be consumed by any of the many runtimes that support ONNX, including Microsoft’s ONNX Runtime.

Next example shows how to export a simple model.


.. autosummary::
   :toctree: generated





ONNX
----
.. automodule:: physicsnemo.deploy.onnx.utils
    :members:
    :show-inheritance:


Physics AI models tend to have layers that are exotic and therefore not supported by ONNX Opsets. The suggested path in that case is to deploy these models as part of PhysicsNeMo.

