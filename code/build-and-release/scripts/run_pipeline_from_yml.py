import os
import argparse
from azureml.core import Workspace, Datastore
from azureml.pipeline.core import Pipeline
from azureml.core.authentication import AzureCliAuthentication


# Parse Definition Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--definition",
    dest='definition',
    type=str,
    help='The name of the ML pipeline YML definition file',
    default='train_pipeline.yml'
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

# Run Pipeline
pipeline_run = pipeline.submit(
    experiment_name="test"
)

# Wait for Pipeline Run Completion
print(pipeline_run.wait_for_completion())
