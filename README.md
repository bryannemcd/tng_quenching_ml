# Applying Random Forest and XGBoost Machine Learning Algorithms to Understand Galaxy Quenching

A repository of work done in preparation for a forthcoming paper entitled "Understanding the regulation of star formation within TNG100 galaxies on
kpc-scales using machine learning I: Global versus local" by Bryanne McDonough, Sathvika S. Iyengar, Ansa Brew-Smith, Asa F.L. Bluck, and Joanna M. Piotrowska. A [pre-print of the paper is available on ArXiv.](https://doi.org/10.48550/arXiv.2604.15182) The code in this repository was developed by Bryanne McDonough, Ansa Brew-Smith, and Sathvika Iyengar. Citation to this work is requested if you use code in this repository.


The `classification_revised` directory contains the random forest and XGBoost classification analyses. The `regression_revised` directory contains the random forest and XGBoost regression analyses. The models for each galaxy subset are included in individual files within the relevant directories. 



The main code for constructing our plots showing the classification and regression results can be found in `pubplots/combined_pubfigs_halosplit.ipynb`. The PCA plot is constructed in `data/pca_global_vs_local.ipynb`.

## Data and Units

The `data` directory includes individual .parquet files with pandas dataframes for the binned annuli of low- and high-mass central and satellite galaxies.  These files are necessary for the regression and classification analyses. The files are created in `Galaxy Partitioning - Annuli Revised.ipynb`. To run that notebook, users would need to download the 'spaxel'-level data from [this Zenodo repository](https://doi.org/10.5281/zenodo.15476126). This data is derived from the TNG100 simulation.

The following table maps the pandas dataframe columns to their meaning and units. Note that only a select number of these columns are used in the analysis presented in our paper.
| Column label | Unit | Notes |
| --- | --- | --- |
| subID | N/A | The unique index into the TNG100 subhalo catalog of the galaxy the annuli belongs to. |
| bin_radii | dimensionless | Galactocentric radial distance of the bin normalized to the galaxy's effective radius, obtained from the Stellar Projected Sizes supplementary catalog.|
| avg_s_mass | $ log_{10}(M_\odot/kpc^2) $ | $\Sigma_*$, the mean stellar mass surface density for the bin|
| avg_SFR | $log_{10}(M_\odot/yr/kpc^2)$ | $\Sigma_{\rm SFR}$, the mean star formation rate surface density for the bin|
| avg_gas_mass | $$log_{10}(M_\odot/kpc^2)$ | $\Sigma_{\rm gas}$, the mean gas mass surface density for the bin|
| quenching_vector | N/A | 0 if the bin is star-forming, 1 if the bin is quenched for the bin|
| gas_mass | $10^{10} M_\odot \; h^{-1}$ | total $M_{gas}$ for the galaxy this bin belongs to, taken from the TNG subhalo catalog |
| s_mass | $10^{10} M_\odot $ (note the lack of $h$) | total $M_{*}$ for the galaxy this bin belongs to, taken from the TNG subhalo catalog |
| BHMass | $10^{10} M_\odot \; h^{-1}$ | mass of the most massive black hole present in the galaxy|
| class | N/A | whether this galaxy is a central (0) or a satellite (1) |
| veldisp | km/s | 1D velocity dispersion of all galaxy member particles, taken directly from TNG subhalo catalog |
| global_SFRs | $M_\odot /yr$  | 20 Myr or 100 Myr time-averaged SFR for a given galaxy, from the Star Formation Rates supplementary catalog  | 
| BH_cumQM | $M_\odot (kpc/Gyr)^2$ | cumulative energy emitted by central black hole in the high-accretion mode, taken from the TNG particle catalogs |
| BH_cumRM | $M_\odot (kpc/Gyr)^2$ | cumulative energy emitted by central black hole in the low-accretion mode, taken from the TNG particle catalogs |
| Mhalo | $10^{10} M_\odot \; h^{-1}$ | mass of FoF halo the bin's subhalo belongs to |
| overdens | dimensionless | galaxy number overdensity measured at the fifth nearest neighbor |
| bulge | dimensionless | The Gini-M20 bulge statistic from the SKIRT Synthetic Images and Optical Morphologies supplementary catalog |
| Bfld | see TNG documentation | SubhaloBfldHalo from the TNG subhalo catalog |
| BH_AGN |  $M_\odot (kpc/Gyr)^2$ | the sum of BH_cumQM and BH_cumRM |
| GrNr | N/A | index into the TNG FoF group catalog to the halo the bin's subhalo belongs to |
