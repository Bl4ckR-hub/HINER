# HINER

This repository provides the official PyTorch implementations for the HINER series of hyperspectral image neural representations.

The code is organized into two subprojects:

| Folder | Description |
| --- | --- |
| [`hiner/`](hiner/) | Implementation of **HINER: Neural Representation for Hyperspectral Image**. It focuses on hyperspectral image compression and downstream classification on compressed HSI. |
| [`hiner++/`](hiner++/) | Implementation of **HINER++ / Compression as Restoration**. It extends implicit HSI representation to compression and restoration tasks, including denoising, inpainting, spatial super-resolution, and spectral super-resolution. |

For installation, datasets, training commands, and task-specific usage, please refer to the README inside each subfolder:

- [`hiner/README.md`](hiner/README.md)
- [`hiner++/readme.md`](hiner++/readme.md)

## Citation

If this repository is useful for your research, please cite the relevant paper.

```bibtex
@inproceedings{shi2024hiner,
  title={HINER: Neural Representation for Hyperspectral Image},
  author={Shi, Junqi and Jiang, Mingyi and Lu, Ming and Chen, Tong and Cao, Xun and Ma, Zhan},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9837--9846},
  year={2024}
}
```

```bibtex
@article{shi2025compression,
  title={Compression as Restoration: A Unified Implicit Approach to Self-Supervised Hyperspectral Image Representation},
  author={Shi, Junqi and Zhang, Qirui and Lu, Ming and Ma, Zhan},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgements

This codebase builds on [HNeRV](https://github.com/haochen-rye/HNeRV) and [SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer). We thank the authors for sharing their implementations.