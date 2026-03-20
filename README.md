# Applying Random Forest and XGBoost Machine Learning Algorithms to Understand Galaxy Quenching

A repository of work done in preparation for a forthcoming paper. The code in this repository was developed by Bryanne McDonough, Ansa Brew-Smith, and Sathvika Iyengar.


The `classification_revised` directory contains the random forest and XGBoost classification analyses. The `regression_revised` directory contains the random forest and XGBoost regression analyses. The models for each galaxy subset are included in individual files within the relevant directories. 

The `data` directory includes individual .parquet files for the binned annuli of low- and high-mass central and satellite galaxies. These files are necessary for the regression and classification analyses. The files are created in `Galaxy Partitioning - Annuli Revised.ipynb`. To run that notebook, users would need to download the 'spaxel'-level data from [this Zenodo repository](https://doi.org/10.5281/zenodo.15476126).

The code associated with constructing each Figure in the associated paper are as follows:
- Figure 1: `classification_revised\rf_classification_annuli\rf_classification_outputs\rf_classification_plots.ipynb`
- Figure 2: `classification_revised\xg_classification_annuli\xg_classification_outputs\xg_classification_plots.ipynb`
- Figure 3: `regression_revised\rf_regression_annuli\rf_regression_outputs\rf_regression_plots.ipynb`
- Figure 4: `regression_revised\xg_regression_annuli\xg_regression_outputs\xg_regression_plots.ipynb`
- Figure 5: `data\pca_global_vs_local.ipynb`