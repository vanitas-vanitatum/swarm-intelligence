import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor

import tqdm

from src.constraints import RoomConstraint
from src.furnishing.furniture_construction import *
from src.furnishing.room_utils import get_example_room

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('number_of_samples', type=int, help='Number of samples to generate')
argument_parser.add_argument('output_path', help='File path to save samples in *.pkl format')
args = argument_parser.parse_args()


def generate_example_solutions(room, constraint, number_of_samples, output_path):
    features = len(room.params_to_optimize)
    features_flatten = len(room.params_to_optimize.flatten())
    boundaries = [(2, room.width - 2), (2, room.height - 2), (-360, 360)] * features
    boundaries = np.array(boundaries)
    correct_solutions = []
    tasks = []
    with ProcessPoolExecutor(max_workers=8) as p:
        for _ in tqdm.tqdm(range(number_of_samples)):
            tasks.append(p.submit(generate_sample, boundaries, features_flatten, constraint))

        for task in tqdm.tqdm(tasks):
            correct_solutions.append(task.result())
    with open(output_path, 'wb') as f:
        pickle.dump(correct_solutions, f, pickle.HIGHEST_PROTOCOL)


def generate_sample(boundaries, features_flatten, constraint):
    sample = np.random.uniform(boundaries[:, 0], boundaries[:, 1], size=(1, features_flatten))
    try:
        while not constraint.check(sample):
            sample = np.random.uniform(boundaries[:, 0], boundaries[:, 1], size=(1, features_flatten))
    except:
        return generate_sample(boundaries, features_flatten, constraint)
    return sample


def main():
    room = get_example_room()
    generate_example_solutions(room, RoomConstraint(room), args.number_of_samples, args.output_path)


if __name__ == '__main__':
    main()
