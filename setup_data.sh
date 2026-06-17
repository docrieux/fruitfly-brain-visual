#!/usr/bin/env bash
# Downloads the two public datasets this kit needs. Run once, from inside
# the fly_brain_sim_kit folder.
set -e

if [ ! -d "Drosophila_brain_model" ]; then
    echo "Cloning Shiu et al. connectome LIF model..."
    git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git
else
    echo "Drosophila_brain_model/ already present, skipping."
fi

if [ ! -d "flywire_annotations" ]; then
    echo "Cloning FlyWire cell-type + 3D position annotations..."
    git clone --depth 1 https://github.com/flyconnectome/flywire_annotations.git
else
    echo "flywire_annotations/ already present, skipping."
fi

echo "Data ready (~280MB total)."
