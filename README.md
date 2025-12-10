# Embracing the Blessing of Smoothness in Optimal Transport Map Estimation Between General Distributions

## Overview
This repository contains code to reproduce results from the paper **“Embracing the Blessing of Smoothness in Optimal Transport Map Estimation Between General Distributions”** by Yizhe Ding, Runze Li, and Lingzhou Xue (The Pennsylvania State University).

---

## Licenses

### Project License (GPL-3.0)

Statistical Convergence Rate of Optimal Transport Map Estimation Between General Distributions is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GNU General Public License](http://www.gnu.org/licenses/) for more details.

### ICNN (Apache-2.0)

This project includes a PyTorch implementation of the Input Convex Neural Network (ICNN), which was originally written in TensorFlow by the authors under the Apache 2.0 License. The original TensorFlow ICNN is licensed under the Apache License 2.0. A copy is included as Apache_LICENSE, and is also available at: http://www.apache.org/licenses/LICENSE-2.0 The original TensorFlow implementation can be  found [here](https://github.com/locuslab/icnn).

## Dependencies
- Python/Numpy
- Scipy
- Sklearn
- PyTorch
- scanpy
- tqdm

## Usage

Train models with different configurations for simulations and real data.

### Reproducing the Simulation Experiments

1) Change into the simulation folder:
2)	Run train.py with the following configuration:

- `--d`: The dimensionality of the training data.
- `--n`: The sample size of the training data.
- `--measure`: The type of probability measure `P`, which can be either `"normal"` or `"t"` (with 6 degrees of freedom as defacult).
- `--transform`: The type of OT map. This should be one of `"CDF"`, `"piecewise_linear"`, or `"quadratic"`.
- `--act`: The activation function of ICNN, which can be one of `"relu"`, `"leaky_relu"`, or `"softplus"`.

**Example**
cd simulation_code
python train.py --d 5 --n 1000 --measure normal --transform CDF --act relu

### Reproduce real data (4i dataset) experiment

1) Download the 4i dataset from https://doi.org/10.3929/ethz-b-000609681
2) Change into the subfolder realdata_code 
3) Run train.py with the chosen activation

- `--act`: The activation function of ICNN, which can be one of `"relu"`, `"leaky_relu"`, or `"softplus"`.

**Example**
cd realdata_code
python train.py --act softplus