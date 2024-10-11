#!/bin/bash

# Check if there are no parameters passed
if [ $# -eq 0 ]; then
    echo "Usage: $0 <parameters>"
    exit 1
fi

# Loop through the parameters and filter out "-Xclang" and replace "/usr/bin/gcc_wrapper" with "/opt/homebrew/bin/g++-12"
filtered_params=""
skip_next=false

for param in "$@"; do
    if [ "$skip_next" = true ]; then
        skip_next=false
        continue
    fi

    if [ "$param" = "-Xclang" ]; then
        skip_next=true
    elif [ "$param" = "./t.sh" ]; then
        filtered_params+=" /opt/homebrew/bin/g++-12"
    else
        filtered_params+=" $param"
    fi
done

$filtered_params
