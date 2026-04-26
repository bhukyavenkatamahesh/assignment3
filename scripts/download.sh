#!/usr/bin/env bash
# Pulls the Soli dataset (preprocessed range-Doppler HDF5 files) into ./data
# Reference: https://github.com/simonwsw/deep-soli  (Wang et al.)

set -e

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://polybox.ethz.ch/index.php/s/wG93iTUdvRU8EaT/download"
ARCHIVE="soli_dsp.zip"

if [ ! -f "$ARCHIVE" ]; then
    echo "downloading soli archive (this can take a while ~10G)"
    curl -L -o "$ARCHIVE" "$URL"
fi

echo "extracting..."
unzip -q -n "$ARCHIVE"
echo "done. files are in $DATA_DIR/"
ls -1 | head
