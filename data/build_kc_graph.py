import os
import json
import pickle
import argparse

import networkx as nx
import pandas as pd


def load_edges(file_path):
    """
    Load KC dependency edges.

    Returns:
        list of tuples:
        [(source, target, weight), ...]
    """

    if file_path.endswith(".csv"):

        df = pd.read_csv(file_path)

        required_columns = [
            "source_kc",
            "target_kc"
        ]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Missing column {col} in csv file"
                )

        edges = []

        for _, row in df.iterrows():

            weight = (
                row["weight"]
                if "weight" in df.columns
                else 1.0
            )

            edges.append(
                (
                    int(row["source_kc"]),
                    int(row["target_kc"]),
                    float(weight)
                )
            )

        return edges


    elif file_path.endswith(".json"):

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)


        edges = []

        for item in data:

            source = int(
                item["source"]
            )

            target = int(
                item["target"]
            )

            weight = float(
                item.get(
                    "weight",
                    1.0
                )
            )

            edges.append(
                (
                    source,
                    target,
                    weight
                )
            )

        return edges


    else:

        raise ValueError(
            "Only csv and json files are supported."
        )


def build_kc_graph(edges):

    """
    Construct KC dependency graph.

    A directed weighted graph is used because
    prerequisite relationships among knowledge concepts
    are directional.
    """

    G = nx.DiGraph()


    for source, target, weight in edges:

        G.add_node(
            source
        )

        G.add_node(
            target
        )

        G.add_edge(
            source,
            target,
            weight=weight
        )


    return G



def save_graph(
        graph,
        output_path
):

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "wb"
    ) as f:

        pickle.dump(
            graph,
            f
        )


def main():

    parser = argparse.ArgumentParser(
        description=
        "Build KC graph for RA4KT"
    )


    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "XES3G5M",
            "ASSISTments2017"
        ]
    )


    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=
        "KC dependency annotation file"
    )


    parser.add_argument(
        "--output",
        type=str,
        default=None
    )


    args = parser.parse_args()


    print(
        f"Building KC graph for {args.dataset}"
    )


    edges = load_edges(
        args.input
    )


    print(
        f"Loaded {len(edges)} KC relations"
    )


    graph = build_kc_graph(
        edges
    )


    print(
        f"Nodes: {graph.number_of_nodes()}"
    )

    print(
        f"Edges: {graph.number_of_edges()}"
    )


    if args.output is None:

        output_path = (
            f"kc_graph/"
            f"{args.dataset.lower()}_graph.pkl"
        )

    else:

        output_path = args.output


    save_graph(
        graph,
        output_path
    )


    print(
        "Saved graph:"
        ,
        output_path
    )


if __name__ == "__main__":

    main()