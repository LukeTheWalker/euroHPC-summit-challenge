#!/bin/bash

# List of implementations
implementations=("cuBlas" "MGPU" "Luca" "Tommy" "OpenMP" "CPU Serial")
orders=(0 1 2 3)

# Compile function
compile() {
    local implementation=${implementations[$order]}
    # Add your compile command here, using the $implementation variable
    # For example: gcc -o program $implementation.c -Wall -Werror
    make clean >/dev/null && make -j4 >/dev/null && ./bin/conj "$1" "$2" "$3" "$4" "$5" $order
}

# Main script
for order in "${orders[@]}"
do
    compile "$1" "$2" "$3" "$4" "$5"
done
