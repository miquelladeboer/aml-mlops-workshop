import argparse
from azureml.core import Workspace
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.model import Model


# Parse Definition Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--name",
    dest='name',
    type=str,
    help='The model name'
)
parser.add_argument(
    "-v",
    "--version",
    dest='version',
    type=int,
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
print("Output dir: ", args.output_dir)

# Connect to the AML Workspace
ws = Workspace.from_config(
    auth=AzureCliAuthentication()
)

# Retrieve the model
model = Model(
    workspace=ws,
    name=args.name,
    version=args.version
)

# Download model to target dir
model.download(
    target_dir=args.output_dir,
    exist_ok=True
)
