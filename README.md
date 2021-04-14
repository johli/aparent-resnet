![APARENT-ResNet Logo](https://github.com/johli/aparent-resnet/blob/master/aparent_resnet_logo.png?raw=true)

# APARENT-ResNet
This repository contains the code for training and running APARENT-ResNet, a deep residual neural network that can predict human 3' UTR Alternative Polyadenylation (APA) and cleavage magnitude at base-pair resolution. This is an improved model compared to the original [APARENT](https://github.com/johli/aparent).

Contact *jlinder2 (at) cs.washington.edu* for any questions about the model or data.

### Web Prediction Tool
We have hosted a publicly accessible web application where users can predict APA isoform abundance and variant effects with APARENT-ResNet and visualize the results.

The web prediction tool is located at [https://apa.cs.washington.edu](https://apa.cs.washington.edu).

### Installation
APARENT-ResNet can be installed by cloning or forking the [github repository](https://github.com/johli/aparent-resnet.git):
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

## Data Availability
The 3' UTR MPRA (the training data and the measured variant data) are available at the original [APARENT GitHub](https://github.com/johli/aparent). For reference, the below link will take you to the data repository.

[Processed Data Repository](https://drive.google.com/open?id=1qex3oY-rarsd7YowM7TxxUklLbLkUyOT)<br/>

The newest version of the data has been re-processed with the following additional improvements compared to the original published version:
1. Exact cleavage positions have been mapped for the Alien1 Random MPRA Sublibrary.
2. A 20 nt random barcode upstream of the USE in the Alien1 Sublibrary has been included in the sequence.

*Note*: The code for the updated data processing is located at the original APARENT GitHub, but was never published.

## Variant Prediction
The following notebook scripts contain benchmark comparison results for the task of classifying disruptive APA variants.

[Notebook: Variant Prediction Benchmark](https://nbviewer.jupyter.org/github/johli/aparent-resnet/blob/master/analysis/apa_variant_prediction_benchmark.ipynb)<br/>
