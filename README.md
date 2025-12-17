## Project Overview
This project analyzes local field potential (LFP) recordings from the mouse auditory cortex
in response to low- and high-frequency tone stimuli.
The analysis pipeline reproduces and refactors the midterm assignment into a modular,
reproducible Python project.

The pipeline includes:
- Loading LFP data from a MATLAB file (`mouseLFP.mat`)
- Baseline-based outlier trial rejection using RMS and peak-to-peak criteria
- 10th-order Butterworth low-pass filtering (cutoff: 1000 Hz)
- Two analysis approaches:
  1. Event-Related Potential (ERP) analysis with peak amplitude/latency and power spectrum
  2. Time-frequency analysis using spectrograms and band-specific power (theta, beta, gamma)
- Saving analysis results and figures for reproducibility

The entire pipeline is executed through a single entry point (`main.py`).

---

## Local approach

### Installation
- Python 3.9 or later
- Required Python packages:
  - numpy
  - scipy
  - matplotlib

### usage examples
python main.py

## Docker approach

### Docker image
https://hub.docker.com/r/yjpark2021/bri519_lfp

### Pull the Docker Image
docker pull yjpark2021/bri519_lfp:latest

### usage examples
docker run --rm -v "$(pwd)/results:/app/results" yjpark2021/bri519_lfp:latest

