#!/bin/bash

fetch() {
    wget https://cdn.intra.42.fr/document/document/34001/datasets.tgz
    tar -xf datasets.tgz
    rm datasets.tgz
}

install() {
    echo "TODO: check uv is installed"

    uv sync

    echo "To activate the venv run : source .venv/bin/activate"
    echo "on windows: .\.venv\Scripts\activate"
}

jupyter() {
    source .venv/bin/activate \
        && python3 -m jupyter notebook
}


launch() {
    uv sync
    echo launching with network: $1
    echo             on dataset: $2

    python split_data.py $2

    echo ""
    echo ""

    python train.py $1 train.csv validation.csv

    echo ""
    echo ""

    MODEL="${1%.txt}.pkl"

    python evaluate.py $MODEL validation.csv
}

usage() {
    cmds=$(declare -F | awk '{print $3}' | paste -s -d'|' -)
    echo "Usage: ./make.sh [$cmds]"
    echo "       ./make.sh launch network.txt data.csv"
}


# ENTRY POINT

cmd="$1"

if [[ -z "$cmd" || "$cmd" == "help" ]]; then
    usage
    exit 0
fi

# Check if function exists, then call
if declare -F "$cmd" > /dev/null; then
    "$cmd" $2 $3
else
    echo "Unknown command: $cmd"
    echo
    usage
    exit 1
fi
