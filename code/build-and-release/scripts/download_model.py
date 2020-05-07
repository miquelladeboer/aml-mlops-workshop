import argparse


# Parse Definition Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    "--name",
    dest='name',
    type=str,
    help='The model name.'
)
parser.add_argument(
    "-v",
    "--version",
    dest='model version',
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
