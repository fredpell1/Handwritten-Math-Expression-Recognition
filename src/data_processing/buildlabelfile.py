"""Given a directory(possibly recursive) of inkml files, will build a file containing the label of each pictures"""
import os
import re
import argparse
from convertfiles import find_all_subdirectories


def find_files(dir_name):
    directories = find_all_subdirectories(dir_name)
    for dir in directories:
        for file in os.listdir(dir):
            if file.endswith('inkml'):
                yield os.path.join(dir, file)


def find_label(file: str):
    regex = r"<annotation type=\"truth\">(?P<label>.*)</annotation>"
    match = re.search(regex, file)
    if match:
        return match.group("label")


def main(src_dir, target_file):
    data = dict()
    for file in find_files(src_dir):
        with open(file, "r") as f:
            data[file] = find_label(f.read())
    with open(target_file, "w") as csvfile:
        for key, item in data.items():
            csvfile.write(f"{key} \",\" {item}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src", default="data\CROHME2016_data\\Task-1-Formula\\TrainINKML"
    )
    parser.add_argument("--target", default="data\\CROHME2016_data\\labels.csv")

    args = parser.parse_args()
    main(args.src, args.target)
