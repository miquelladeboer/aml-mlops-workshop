from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model
from azureml.core import Workspace

import os

ws = Workspace.from_config()
model_name = "net.onnx"

model = Model(workspace=ws, name=model_name)

myenv = CondaDependencies(
    conda_dependencies_file_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../',
        'conda_dependencies.yml'
    )
)
myenv.add_channel("pytorch")

with open("myenv.yml", "w") as f:
    f.write(myenv.serialize_to_string())


myenv = Environment.from_conda_specification(name="myenv",
                                             file_path="myenv.yml")

inference_config = InferenceConfig(
    entry_script="code_final/deployment/score.py",
    environment=myenv)


aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               tags={'demo': 'onnx'},
                                               description='ONNX for text')


aci_service_name = 'onnx-demo2'
print("Service", aci_service_name)
aci_service = Model.deploy(ws, aci_service_name, [model],
                           inference_config, aciconfig)
aci_service.wait_for_deployment(True)
print(aci_service.state)

if aci_service.state != 'Healthy':
    # run this command for debugging.
    print(aci_service.get_logs())

print(aci_service.scoring_uri)

