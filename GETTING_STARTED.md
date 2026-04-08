# CycleGAN and pix2pix PyTorch Implementation

## Overview

This repository contains PyTorch implementations of CycleGAN and pix2pix models for unpaired and paired image-to-image translation. The code supports:

- **CycleGAN**: Unpaired image-to-image translation (e.g., photo to painting, day to night)
- **pix2pix**: Paired image-to-image translation (e.g., sketch to photo, edges to cats)
- **Single Model**: Generate results for one image
- **Colorization**: Convert grayscale images to color

## Quick Start

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate pytorch-img2img
```

### Training

**CycleGAN on maps dataset:**
```bash
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

**pix2pix on facades dataset:**
```bash
python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix
```

### Testing

**Test a trained CycleGAN model:**
```bash
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
```

## Project Structure

```
├── data/               # Data loading and preprocessing
├── models/             # Model implementations
├── options/            # Training/test options
├── util/               # Utility functions
├── datasets/           # Dataset download scripts
├── scripts/            # Training/test shell scripts
├── docs/               # Documentation
└── checkpoints/        # Trained models (created during training)
```

## Key Features

- Supports PyTorch 2.4+
- DDP (Distributed Data Parallel) training with `torchrun`
- M-series Mac support via MPS
- Integration with Weights & Biases (wandb) for logging
- Customizable datasets and models

## Documentation

- [Model Overview](docs/overview.md)
- [Training Tips](docs/tips.md)
- [FAQ](docs/qa.md)
- [Docker Setup](docs/docker.md)

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{zhu2017CycleGAN,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}

@inproceedings{isola2017pix2pix,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
}
```

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original CycleGAN paper by Jun-Yan Zhu et al.
- Original pix2pix paper by Phillip Isola et al.
- PyTorch community for excellent tools and libraries
