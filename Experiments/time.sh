
#!/bin/bash

functions=("NN" "Hyperplane" "Hypersphere" "SingleDimension" "Hyperbolic" "Ellipsoid" "Paraboloid" "Quadric")

for function in "${functions[@]}"; do
    /usr/local/bin/python3 Yourpath/FuBIF/Experiments/time_experiment.py --model "$function"
done