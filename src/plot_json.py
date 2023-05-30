"""Generate Image from Openpose JSON

Returns:
    .jgp: Skeleton Image

Source: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1644#issuecomment-657361528

Run from Terminal using:
python -m plot_json ./images/1lady.jpg.json ./images/OpenPoseGT/output.jpg 1000 1000
python -m plot_json ./images/OpenPoseGT/A-pose_keypoints_GT.json ./images/OpenPoseGT/ou
tput.jpg 1500 1500
"""

import sys
from json import load
from math import ceil

import click
import gizeh
from more_itertools.recipes import grouper, pairwise

# From https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/20d8eca4b43fe28cefc02d341476b04c6a6d6ff2/doc/output.md#pose-output-format-body_25
BODY_25_LINES = [
    [17, 15, 0, 1, 8, 9, 10, 11, 22, 23],  # Right eye down to right leg
    [11, 24],  # Right heel
    [0, 16, 18],  # Left eye
    [4, 3, 2, 1, 5, 6, 7],  # Arms
    [8, 12, 13, 14, 20],  # Left leg
    [14, 21]  # Left heel
]


def build_graph(lines):
    graph = {}
    for line in lines:
        for n1, n2 in pairwise(line):
            if n1 > n2:
                n1, n2 = n2, n1
            graph.setdefault(n1, set()).add(n2)
    return graph


BODY_25_GRAPH = build_graph(BODY_25_LINES)


def max_dim(doc, dim):
    return max((
        val
        for person in doc["people"]
        for numarr in person.values()
        for val in numarr[dim::3]
    ))


@click.command()
@click.argument("jsonin", type=click.File("r"))
@click.argument("pngout", type=click.File("wb"))
@click.argument("width", type=int, required=False)
@click.argument("height", type=int, required=False)
def plot_OpenposeJSON(jsonin: str, pngout: str, width: int, height: int):
    doc = load(jsonin)
    if not width or not height:
        print("Warning: no width/height specified. Setting to max known + 10.", file=sys.stderr)
        width = ceil(max_dim(doc, 0)) + 10
        height = ceil(max_dim(doc, 1)) + 10
    surface = gizeh.Surface(width=width, height=height, bg_color=(1, 1, 1))
    for person in doc["people"]:
        numarr = list(grouper(person["pose_keypoints_2d"], 3))
        for idx in range(len(numarr)):
            for other_idx in BODY_25_GRAPH.get(idx, set()):
                x1, y1, c1 = numarr[idx]
                x2, y2, c2 = numarr[other_idx]
                c = min(c1, c2)
                if c == 0:
                    continue
                line = gizeh.polyline(
                    points=[(x1, y1), (x2, y2)], stroke_width=5 * c,
                    stroke=(0, 0, 0)
                )
                line.draw(surface)

    surface.write_to_png(pngout)


if __name__ == "__main__":
    plot_OpenposeJSON()