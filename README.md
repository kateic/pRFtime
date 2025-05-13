# pRFtime: a spatiotemporal forward modeling approach of human brain responses 

**pRFtime** is a biologically-inspired forward modeling framework for analyzing spatiotemporal brain responses. It combines population receptive field (pRF) modeling, stimulus-based cortical predictions, and forward modeling to quantify the contributions of brain regions to the measured signals over time. 

> This framework is designed for flexibility, reproducibility, across modalities and neuroscientific fields. 

For a full application see [*Eickhoff et al. (2025), bioRxiv*](https://www.biorxiv.org/content/10.1101/2025.05.12.653426v1), 
A publication for a in depth explanation of the pipeline flow and algorithms used will follow soon. For now, please cite the application paper (Eickhoff et al. (2025), *bioRxiv*) when using the code. 

## Test Data for pRFtime 
Due to file size constraints, the data are not included in the GitHub repository, but are stored and publicly available on the OSF website. 
To download the the test data: Go to https://osf.io/hnw5d/ and download all files in the project. The files should be stored in the `pRFtime/test/example_data` folder on your system. The data files together are 3.7GB.  
The test data files contain a sample from the dataset used in Eickhoff et al. (2025), *bioRxiv*, allowing you to replicate the results and get familiar with the pipeline.  Substituting data with your own data is encouraged. 

## Features 
- Full pipeline from pRFs to variance explained time-courses of brain regions of interest with high temporal resolution 
- Modular: customize pRF models, stimuli, regions of interest, and gain matrices 
- Cross-validated regularized regression procedure to ensure robust model fitting 
- Fully implemented in Python with open-source tools 

## Installation 
Clone the repository and install dependencies: 

```bash
git clone https://github.com/kateic/pRFtime.git
cd pRFtime 
python setup.py develop

``` 
Dependencies / python packages: 

```
os, random, math, numpy, sklearn, matplotlib, jupyterlab, networkx, seaborn, fracridge
```

## Example Usage 
Check out the example notebook: `pRFtime/run_pipeline.ipynb`

This notebooks walks through: 
1. Loading the test data you've downloaded 
2. Creating sensor-level predictions of regions of interset (ROIs) based on pRF models 
3. Fitting sensor predictions to measured sensor data 
4. Visualizing model performance and ROI contributions to the measured signal 
5. An example of model validation procedures 

## Project Structure 
```
pRFtime/

|--- pRFtime/
    |--- core.py                        # Core classes and modeling functions 
    |--- utils.py                       # Helper functions and pRF Model classes 
    |--- figures.py                     # Helper functions to plot Test data output in `run_pipeline.ipynb` 
|--- test/
    |--- example_data/                  # Create this folder and download the test data from OSF into this folder - the files included are:  
        |--- example_prf_parameters.npy 
        |--- example_sensor_data.npy     
        |--- example_design_matrix.npy   
        |--- example_gain_matrix.npy      
        |--- example_roi_masks.npy       
        |--- example_mask.npy            

|--- README.md                          # You are here 
|--- run_pipeline.ipynb                 # Text Jupyter notebook 
```

## Inputs and Outputs 
* **Inputs**: 
    > * pRF model parameters (e.g. 2D Gaussians)
    > * Stimulus design matrix 
    > * Gain matrix (e.g. based on Overlapping Spheres model)
    > * Sensor data (e.g. MEG recordings to stimuli in design matrix)
    > * Region of interest (ROI) masks 
    > * optional mask for exclusion of vertices (cortical locations)
* **Outputs**: 
    > * Full-model variance explained over time 
    > * Individual ROI contributions over time 
    > * Optimized regularization parameters 
    > * Diagnostic plots (optional)

## Citation 
If you use this software in your work, please cite: 

> Eickhoff, K., Hillebrand, A., Knapen, T., de Jong, M. C., & Dumoulin, S. O. (2025). Non-invasive mapping of the temporal processing hierarchy in the human visual cortex. bioRxiv. https://doi.org/10.1101/2025.05.12.653426

A software publication detailing the algorithms will follow soon.

## License 
**pRFtime** is licensed under the terms of the GPL-v3 license (see LICENSE file for more information).



Katharina Eickhoff, 2025, Spinoza Centre for Neuroimaging, Amsterdam. 
