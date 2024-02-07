# n5-dask

Dask-based utilities to process N5 images. This repo is a proof of concept and is not meant to be used for production purposes.

## Usage

1. Load the dependencies

```
conda env create --file environment.yml
conda activate n5-dask
```

2. Run any of the scripts

```
mkdir output
python src/n5_dask/n5_to_tif.py -i data/test.n5/ -d mri/c0/s0 -o output
```

