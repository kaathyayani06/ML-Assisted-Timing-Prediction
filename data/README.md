# Dataset Setup

This project uses the **8_rat dataset** for graph-based timing prediction.

## Dataset Source

The dataset is based on the TimingPredict dataset:

[TimingPredict Dataset – GitHub](https://github.com/TimingPredict/Dataset)

## Dataset Information

The dataset contains graph data for ten benchmark circuits.

### Training Circuits
- aes_cipher
- blabla
- des
- genericfir
- spm
- usb

### Testing Circuits
- BM64
- cic_decimator
- synth_ram
- zipdiv

## Directory Structure

After obtaining the required processed graph files, place them in the following directory:

```text
data/
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

## Important Notes

- The complete dataset is approximately 4.14 GB.
- Dataset files are not included in this GitHub repository because of their size.
- Users must obtain the required graph files separately and place them in the directory structure shown above.
- The training and evaluation scripts expect the graph files to be available under `data/8_rat/`.

