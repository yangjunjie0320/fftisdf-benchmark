# FFT-ISDF Benchmark

## Dependencies

The dependencies are listed in `environment.yml`. You can install them by running:

```bash
conda env create -f environment.yml
```

and activate the environment by running:

```bash
conda activate fftisdf
```

## Crystal Structure
All the crystal structures used in this benchmark are stored in `data/crystal_structures`. They are all downloaded from the Materials Project
with the `mp_api` package (refer to `/src/build.py` for more details).

