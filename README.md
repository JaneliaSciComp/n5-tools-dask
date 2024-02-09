# n5-tools-dask

Dask-based utilities to process N5 images. This repo is a proof of concept and is not meant to be used for production purposes.

## Usage

1. Load the dependencies

```
conda env create --file environment.yml
conda activate n5-tools-dask
```

2. Run any of the scripts

```
mkdir output_tiffs output_n5
python src/n5_to_tif.py -i data/test.n5/ -d mri/c0/s0 -o output_tiffs
python src/tif_to_n5.py -i output_tiffs -o output_n5 -d mri/s0 -c 64,64,64 --compression gzip
python src/n5_multiscale.py -i output_n5 -d mri
```

This will reprocess the included n5 into a series of TIFFs, which are then reprocessed back into an n5 with smaller chunking. Finally, it will add multiscale imagery.

