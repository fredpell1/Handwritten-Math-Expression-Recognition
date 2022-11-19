"""Convert all the inkml files in the passed directory and its subdirectories in parallel"""
import os
import subprocess
import argparse

def find_all_subdirectories(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    for directory in list(subfolders):
        subfolders.extend(find_all_subdirectories(directory))
    return subfolders


def main(in_directory=None, out_directory=None, dim=None, padding=None):
    subdirectories = find_all_subdirectories(in_directory)
    processes = [
        subprocess.Popen(
            [
                "python",
                "src\data_processing\convertInkmlToImg.py",
                dir,
                dim,
                padding,
                out_directory,
            ],
            shell=True,
        )
        for dir in subdirectories
    ]
    for proc in processes:
        proc.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', default='data\Task-2-Symbols')
    parser.add_argument('--outdir', default='data\\data_png\\')
    parser.add_argument('--dim', default='300')
    parser.add_argument('--padding', default='2')

    args = parser.parse_args()

    main(
        in_directory=args.indir,
        out_directory=args.outdir,
        dim=args.dim,
        padding=args.padding,
    )
