# Dataset

This directory is intended to contain the dataset used for the ML-Assisted Timing Prediction project.

## Dataset Source

The experimental dataset is obtained from the open-source TimingPredict project.

**Source:** https://github.com/TimingPredict/Dataset

## Dataset Details

- Dataset folder: `8_rat`
- Approximate size: 4.14 GB
- Contains the complete dataset used in the experiments.
- Includes benchmark circuit data for training and testing.

## Benchmark Circuits

**Training Circuits:**
- aes_cipher
- blabla
- des
- genericfir
- spm
- usb

**Testing Circuits:**
- BM64
- cic_decimator
- synth_ram
- zipdiv

## Local Directory Structure

The complete dataset is maintained locally and is not uploaded to this GitHub repository because of its large size.

Place the `8_rat` folder inside the project's `data/` directory when preparing the local experimental environment.

```text
ML-Assisted-Timing-Prediction/
└── data/
    ├── README.md
    └── 8_rat/
        ├── aes_cipher/
        ├── blabla/
        ├── des/
        ├── genericfir/
        ├── spm/
        ├── usb/
        ├── BM64/
        ├── cic_decimator/
        ├── synth_ram/
        └── zipdiv/
```
