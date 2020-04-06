import os
from azureml.core import Workspace, Model
from azureml.core.authentication import AzureCliAuthentication

ws = Workspace.from_config(
    auth=AzureCliAuthentication()
)

model = Model(
    workspace=ws,
    name="net.onnx"
)

model.download(
    target_dir=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../',
        'outputs'
    )
)

print(model)
