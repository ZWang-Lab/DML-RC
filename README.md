# DML-RC
Double/debiased Machine Learning with Regression Calibration (DML-RC) is a machine learning approach to estimate the causal effects of correlated multi-pollutant and correct for bias due to measurement error. 

## Installation Instructions:

### Required software and packages
    
1. Python 3.8 or higher
    
2. Package:    numpy, pandas, math, pickle, time, statsmodels, scipy, copy, doubleml, sklearn, multiprocessing
    
3. Python code:   minimax_tilting_sampler.py (directly download from https://github.com/brunzema/truncated-mvn-sampler)

Please install the required packages and codes before you use DML-RC code.

## Usage instructions

Directly download generate_data.py and reg_dml.py, then run the following code in Python:

```
import generate_data
import reg_dml
```
Example code (example.ipynb) was provided to simulate dataset and estimate causal effect with DML-RC.
