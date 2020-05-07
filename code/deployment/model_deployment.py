import os
import argparse
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice
from azureml.core.model import Model
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication


# Parse Definition Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--name",
    dest='name',
    type=str,
    default='net.onnx',
    help='The model name'
)
parser.add_argument(
    "-v",
    "--version",
    dest='version',
    type=int,
    default=3,
    help='The model version'
)
parser.add_argument(
    "-o",
    "--output",
    dest='output_dir',
    type=str,
    help='output folder'
)
args = parser.parse_args()

print("Model name: ", args.name)
print("Model version: ", args.version)

# Load the AML Workspace and Model
ws = Workspace.from_config(
    auth=AzureCliAuthentication()
)

model = Model(
    workspace=ws,
    name=args.name,
    version=args.version
)

# Configure Scoring Environment
# myenv = CondaDependencies(
#     conda_dependencies_file_path=os.path.join(
#         os.path.dirname(os.path.realpath(__file__)),
#         '../../',
#         'conda_dependencies.yml'
#     )
# )
# myenv.add_channel("pytorch")

# with open("myenv.yml", "w") as f:
#     f.write(myenv.serialize_to_string())

scoringenv = Environment.from_conda_specification(
    name="scoringenv",
    file_path=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../../environments/',
        'scoring/conda_dependencies.yml'
    )
)

# Configure Service Deployment Enviromment and compute-wise
service_name = 'onnx-demo'

inference_config = InferenceConfig(
    entry_script=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        '../modeling/score.py'
    ),
    environment=scoringenv
)

compute_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    tags={
        'demo': 'onnx'
    },
    description='ONNX for text'
)

# Run the deployment
deployment = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=compute_config
)

# Wait for completion
deployment.wait_for_deployment(True)

# Print debug if deployment was unsuccesful
if deployment.state != 'Healthy':
    # run this command for debugging.
    print(deployment.get_logs())

    # thow error, is this done?

print(deployment.scoring_uri)
