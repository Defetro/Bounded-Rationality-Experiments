import os
import json
import argparse

from agent import Agent

def main():
    args = parse_command_line_args()
    config = load_config(args.experiment)
    A = Agent(config)
    A.run()


def load_config(experiment):
    with open("./config/" + str(experiment) + ".json") as json_file:
        config = json.load(json_file)
    return config


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment',
                        help='The json path containing the target experiment configuration.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()