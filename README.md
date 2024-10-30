# Knowledge graph-primed deep learning to identify condition-specific gene importance

This repository contains the code for the master's thesis Knowledge graph-primed deep learning to identify condition-specific gene importance.
For running, it requires the knowledge network Comprehensive Knowledge Network and a gene expression dataset that has been annotated with tissue types or perturbation groups.

## Files
### Data preprocessing

The following files are used to preprocess the data :
- *dee2_proc.R* : used to convert DEE2 Kallisto counts into a TPM table,
- *unlabeled_data_preparation.py* : trims the DEE2 TPM table according to the quality control values,
- *combine_with_metadata.py* : combines the gene expression data subset with annotations,
- *load_tpm_data.py* : contains the dataloader used for prediction model training/evaluation,
- *subset_data.py* : creates a subset from the annotated data, limiting the samples to a set of perturbation groups,
- *utils.py* : contains two functions, *generate_isoform_file* and *average_isoforms_in_data*, that handle the gene isoforms.

### Batch effect correction
The following files are used to remove the batch effect from the preprocessed data:
- *combat.R* : utilises ComBat and ComBat-Seq,
- *limma.R* : utilises the Limma library,
- *scvi_VAE_colab_notebook.ipynb* : the Google Colab notebook used for scVI VAE training,
- *batch_effect_measurement.py* : calculates the clustering-based metric on the batch effect corrected data,
- *evaluate_be_correction.py* : computes the BEC metric on a series of experiments,
- *tsne_plotting.py* : uses t-SNE plots to visualise the result of batch effect correction.

### Prediction models
The following files are used to train and evaluate the prediction models:
- *training.py* : contains the class used to train a single model,
- *predict.py* : contains the class used to evaluate a single model,
- *CKN_make_parallel_layers.py* : creates the files used for CKN-based model,
- *run_experiments.py* : used to train and evaluate a series of models, including cross-validation,

### Model interpretation
The following file interprets the trained prediction models :
- *model_interpretation.py* : computes relevant genes from previously trained models.

### Various files
The following files were used for data analysis and plot generation :
- *analyse_batches.py* : analyses the batch distribution,
- *isoform_analysis.py* : analyses the isoform distribution,
- *plot_tissue_distribution.py* : analyses the distribution of tissues within perturbation groups and vice versa,
- *global_values.py* : contains the predefined data paths.



## Running the pipeline

- First, the data is preprocessed. *combine_with_metadata.py* is run, if only joining TPM with metadata is necessary,
- Once the data is preprocessed, one of the batch effect removal methods is chosen,
- The batch effect corrected data is used in *run_experiments.py* to generate all the variations of our prediction models,
- The chosen prediction models are interpreted using *model_interpretation.py*
