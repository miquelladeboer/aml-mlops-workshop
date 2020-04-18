import os
import argparse
from azureml.core import Workspace, Datastore
from azureml.pipeline.core import Pipeline
from azureml.core.authentication import AzureCliAuthentication


# Parse Definition Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--name",
    dest='name',
    type=str,
    help='The name of the published ML pipeline.'
)
parser.add_argument(
    "-d",
    "--definition",
    dest='definition',
    type=str,
    help='The name of the ML pipeline YML definition file.'
)
args = parser.parse_args()

# Load Pipeline from YML
pipeline_definition_path=os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../pipelines',
    args.definition
)

ws = Workspace.from_config(
    auth=AzureCliAuthentication()
)

pipeline = Pipeline.load_yaml(
    ws,
    pipeline_definition_path
)

print("Parsed pipeline graph from YML {}".format(pipeline.graph))

# Publish Pipeline (Endpoint)
published = pipeline.publish(
    name=args.name
)

print("Published YML pipeline {}".format(published))