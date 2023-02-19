import argparse
import os

# matplotlib.use("Agg")  # Needed to run without X-server
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        default=[],
        type=str,
        help="specify paths of scores.txt",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=[],
        type=str,
        help="specify labels for scores.txt files",
    )
    parser.add_argument("--save", dest="save", default=False, type=bool)
    args = parser.parse_args()

    assert len(args.files) > 0
    assert len(args.labels) == len(args.files)

    for fpath, label in zip(args.files, args.labels):
        if os.path.isdir(fpath):
            fpath = os.path.join(fpath, "scores.txt")
        assert os.path.exists(fpath)
        scores = pd.read_csv(fpath, delimiter="\t")
        # plt.plot(scores["steps"], scores["mean"], label=label)
        plt.plot(
            scores["episodes"], scores["mean"], label=label, color="yellowgreen"
        )
        # plt.plot(scores["episodes"], scores["max"], label="max")

    # plt.xlabel("steps")
    plt.xlabel("episodes")
    plt.ylabel("total reward")
    plt.legend(loc="best")
    if args.title:
        plt.title(args.title)

    fig_fname = args.files[0] + "mean" + ".png"
    fig_fname_svg = args.files[0] + "mean" + ".svg"
    if args.save:
        plt.savefig(fig_fname)
        plt.savefig(fig_fname_svg)
    else:
        plt.show()
    print("Saved a figure as {}".format(fig_fname))


if __name__ == "__main__":
    main()
