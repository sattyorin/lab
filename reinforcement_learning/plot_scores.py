import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    current_script_path = os.path.abspath(__file__)
    directories = glob.glob(
        os.path.join(os.path.dirname(current_script_path), "results", "*/")
    )
    latest_directory = max(directories, key=os.path.getctime)

    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--file",
        default=os.path.join(
            "results", os.path.basename(os.path.normpath(latest_directory))
        ),
        type=str,
        help="specify path of scores.txt",
    )
    parser.add_argument("--save", dest="save", default=False, type=bool)
    args = parser.parse_args()

    score_path = os.path.join(args.file, "scores.txt")
    assert os.path.exists(score_path)
    scores = pd.read_csv(score_path, delimiter="\t")
    plt.plot(
        scores["episodes"].to_numpy(),
        scores["mean"].to_numpy(),
        label="mean",
        color="pink",
        ls="-",
    )
    plt.plot(
        scores["episodes"].to_numpy(),
        scores["max"].to_numpy(),
        label="max",
        color="yellowgreen",
        ls="--",
    )

    plt.xlabel("episodes")
    plt.ylabel("total reward mean")
    plt.legend(loc="best")
    if args.title:
        plt.title(args.title)

    fig_fname_png = args.file + "mean" + ".png"
    fig_fname_svg = args.file + "mean" + ".svg"
    if args.save:
        plt.savefig(fig_fname_png)
        plt.savefig(fig_fname_svg)
        print("Saved a figure as {}".format(fig_fname_png))
        print("Saved a figure as {}".format(fig_fname_svg))
    else:
        plt.show()


if __name__ == "__main__":
    main()
