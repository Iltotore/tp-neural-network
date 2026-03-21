#!/bin/bash

input=$(cat -)

echo "${input}" | gnuplot -p -e "plot '-' using 1:2 title '% error by iteration (batch of )' with lines"