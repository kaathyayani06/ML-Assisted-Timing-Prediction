# Dataset

## Dataset Source

The benchmark dataset used in this project is obtained from the open-source TimingPredict project.

**Source:** https://github.com/TimingPredict/Dataset

## Dataset Overview

- Dataset directory: `8_rat`
- Approximate size: 4.14 GB
- Contains benchmark data for ten circuits.
- Used for GNN training, MLP baseline training, and cross-circuit evaluation.

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

## Expected Directory Structure

The dataset is maintained locally and is not uploaded to this repository because of its large size.

The project code expects the following files inside the `8_rat` directory:

```text
data/
├── README.md
└── 8_rat/
    ├── aes_cipher.graph.bin
    ├── blabla.graph.bin
    ├── des.graph.bin
    ├── genericfir.graph.bin
    ├── spm.graph.bin
    ├── usb.graph.bin
    ├── BM64.graph.bin
    ├── cic_decimator.graph.bin
    ├── synth_ram.graph.bin
    └── zipdiv.graph.bin
```

**Note:** The directory structure above represents the filenames expected by the project code. The complete dataset remains available locally.
