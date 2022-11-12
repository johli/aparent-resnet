[![DOI](https://www.zenodo.org/badge/357736760.svg)](https://www.zenodo.org/badge/latestdoi/357736760)

![APARENT2 Logo](https://github.com/johli/aparent-resnet/blob/master/aparent_resnet_logo.png?raw=true)

# APARENT2
This repository contains the code for training and running APARENT2, a deep residual neural network that can predict human 3' UTR Alternative Polyadenylation (APA) and cleavage magnitude at base-pair resolution. This is an updated model compared to the original [APARENT](https://github.com/johli/aparent).

Contact *jlinder2 (at) stanford.edu* for any questions about the model or data.

### Web Prediction Tool
We have hosted a publicly accessible web application where users can predict APA isoform abundance and variant effects with APARENT2 and visualize the results.

The web prediction tool is located at [https://apa.cs.washington.edu](https://apa.cs.washington.edu).

### Installation
APARENT2 can be installed by cloning or forking the [github repository](https://github.com/johli/aparent-resnet.git):
```sh
git clone https://github.com/johli/aparent-resnet.git
cd aparent-resnet
python setup.py install
```

#### APARENT requires the following packages to be installed
- Python >= 3.6
- Tensorflow == 1.13.1
- Keras == 2.2.4
- Scipy >= 1.2.1
- Numpy >= 1.16.2
- Isolearn >= 0.2.0 ([github](https://github.com/johli/isolearn.git))
- [Optional] Pandas >= 0.24.2
- [Optional] Matplotlib >= 3.1.1

### Example: Scoring Variants with APARENT2
The following notebook demonstrates how to use the APARENT2 model to score polyadenylation signal variants: [aparent2_score_variants.ipynb](https://github.com/johli/aparent-resnet/blob/master/examples/aparent2_score_variants.ipynb).

### Genome-wide In-silico Saturation Mutagenesis (Human)
The following google drive folder contains in-silico saturation mutagenesis predictions for all polyadenylation signals found in PolyADB V3 (transcript-wide). The file 'aparent2_ism_scores_polyadb_v3.csv.gz' contains all data. The file 'aparent2_ism_scores_polyadb_v3_cutoff.csv.gz' contains only variants with more than 1.25-fold increase or decrease in isoform odds. The data columns 'delta_logodds' and 'delta_usage' contain variant isoform log odds ratios and isoform proportion differences (wrt. PolyADB measurements) for polyadenylation occurring anywhere +/- 100bp of the canonical cleavage site. The columns 'delta_logodds_narrow' and 'delta_usage_narrow' contains log odds ratios and proportion differences for cleaveage that occurs +0bp to +50bp immediately downstream of the canonical core hexamer motif. The data columns 'pas_position_hg19' and 'pas_position_hg38' indicate the start coordinate of the core hexamer.

[APARENT2 ISM Scores](https://drive.google.com/open?id=1rg7VHKBM19iFIruzDgQ4BtUVqgcjypxu)<br/>

## Data Availability
The 3' UTR MPRA (the training data and the measured variant data) are available at the original [APARENT GitHub](https://github.com/johli/aparent). For reference, the below link will take you to the data repository.

[Processed Data Repository](https://drive.google.com/open?id=1qex3oY-rarsd7YowM7TxxUklLbLkUyOT)<br/>

The newest version of the data has been re-processed with the following additional improvements compared to the original published version:
1. Exact cleavage positions have been mapped for the Alien1 Random MPRA Sublibrary.
2. A 20 nt random barcode upstream of the USE in the Alien1 Sublibrary has been included in the sequence.

*Note*: The code for the updated data processing is located at the original APARENT GitHub, but was never published.

## Notebooks
The following notebook scripts contain benchmark comparison results for various prediction tasks as well as other analyses performed in the paper.

[Notebook 1: (Bogard et al) MPRA Variant Prediction Benchmark](https://nbviewer.jupyter.org/github/johli/aparent-resnet/blob/master/analysis/seelig_variants/apa_variant_prediction_benchmark.ipynb)<br/>
