# HINER++: Compression as Restoration

Official PyTorch implementation of **Compression as Restoration: A Unified Implicit Approach to Self-Supervised Hyperspectral Image Representation**.

HINER++ models a hyperspectral image (HSI) as an implicit neural representation: each spectral band is reconstructed from its wavelength index by a compact neural network. The same self-supervised framework supports HSI compression and restoration tasks such as denoising, inpainting, spatial super-resolution, and spectral super-resolution.

## Highlights

- Unified implicit representation for HSI compression and restoration.
- Self-supervised training on a single HSI sample.
- Instance-specific neural representation with configurable model capacity.
- Support for `.mat` and `.tif` hyperspectral inputs.
- Optional model and embedding quantization for compression experiments.

## Repository Layout

```text
.
|-- data/                 # HSI dataset loaders
|-- example/              # Training and evaluation entry points
|-- models/               # HINER, HINER++, layers, and ViT modules
|-- quantization/         # Quantization and Huffman coding utilities
|-- hnerv_utils.py        # Training, metrics, losses, and visualization helpers
`-- requirements.txt
```

## Installation

Create an environment with Python 3.8 or newer, then install PyTorch for your CUDA version from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

```bash
conda create -n hiner python=3.8
conda activate hiner

# Example only. Pick the PyTorch command that matches your CUDA/CPU setup.
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

pip install -r requirements.txt
```

## Data Format

Input hyperspectral data can be provided as:

- `.mat`: the last variable in the file is loaded as the HSI tensor.
- `.tif`: loaded with `skimage.io.imread` and interpreted as spectral bands by height by width.

Most examples expect `--ori_shape H_W_C`, where `H` and `W` are spatial dimensions and `C` is the number of spectral bands, for example `696_520_34`.

## Quick Start

The commands below are configured for the realistic noisy HSI dataset from [HSIDwRD](https://github.com/ColinTaoZhang/HSIDwRD/tree/main), where each sample has an original shape of `696 x 520 x 34`.

For other datasets, update dataset-dependent parameters such as `--crop_list`, `--ori_shape`, `--fc_hw`, `--dec_strds`, and `--data_norm`.

Key rules:

- `--ori_shape` should match the original HSI shape as `H_W_C`.
- `--crop_list` is the padded/cropped spatial size used by the model. It should be compatible with the decoder output size.
- For HINER++, the reconstructed spatial size is approximately `fc_hw * product(dec_strds)`. For example, `--fc_hw 4_3` and `--dec_strds 5 3 3 2 2` give `4*5*3*3*2*2 = 720` and `3*5*3*3*2*2 = 540`, so `--crop_list 720_540`.
- `--data_norm` should match the bit depth or intensity scale of the data. For 12-bit HSI data, `4096` is commonly used; use `-1` for per-band min-max normalization, which may lead to better results.
- Spatial SR may use different decoder strides. For example, the 4x spatial SR command uses `--dec_strds 5 3 3 1 1`, which reconstructs the low-resolution input before the model upsamples internally.

### 1. Regression / Neural Representation

Train an implicit neural representation for one HSI sample.

<details>
<summary>Show command</summary>

```bash
python example/regression.py \
    --outf HINER \
    --data_path /path/to/hsi.tif \
    --vid sample_regression \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --dec_strds 5 3 3 2 2 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --modelsize 0.5 \
    -e 300 \
    --eval_freq 30 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001 \
    --dump_images
```

</details>

### 2. Real Denoising

Train with a clean/noisy HSI pair.

<details>
<summary>Show command</summary>

```bash
python example/real_denoise.py \
    --outf HINER \
    --clean_data_path /path/to/clean.tif \
    --noise_data_path /path/to/noisy.tif \
    --vid sample_real_denoise \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --dec_strds 5 3 3 2 2 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --modelsize 1.5 \
    -e 150 \
    --eval_freq 30 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001 \
    --dump_images
```

</details>

### 3. Synthetic Restoration

Use `example/restore.py` for synthetic denoising and inpainting.

#### 3.1 Denoising

Supported strengths: `denoise_0.05`, `denoise_0.1`, `denoise_0.15`, `denoise_0.2`.

<details>
<summary>Show command</summary>

```bash
python example/restore.py \
    --outf HINER \
    --data_path /path/to/hsi.tif \
    --vid sample_synthetic_denoise \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --dec_strds 5 3 3 2 2 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --restore denoise_0.1 \
    --modelsize 1.5 \
    -e 300 \
    --eval_freq 30 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001 \
    --dump_images
```

</details>

#### 3.2 Inpainting

Supported masks: `inpaint_text`, `inpaint_block`, `inpaint_hline`.

<details>
<summary>Show command</summary>

```bash
python example/restore.py \
    --outf HINER \
    --data_path /path/to/hsi.tif \
    --vid sample_inpainting \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --dec_strds 5 3 3 2 2 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --restore inpaint_text \
    --modelsize 1.5 \
    -e 300 \
    --eval_freq 30 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001 \
    --dump_images
```

</details>

### 3.3 Spatial Super-Resolution

Supported ratios: `spatialSR_2`, `spatialSR_3`, `spatialSR_4`.

<details>
<summary>Show command</summary>

```bash
python example/spatial_sr.py \
    --outf HINER \
    --data_path /path/to/hsi.tif \
    --vid sample_spatial_sr \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --dec_strds 5 3 3 1 1 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --restore spatialSR_4 \
    --modelsize 1.5 \
    -e 300 \
    --eval_freq 5 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001
```

</details>

### 3.4 Spectral Super-Resolution

Supported ratios: `2`, `3`, `4`.

<details>
<summary>Show command</summary>

```bash
python example/spectra_sr.py \
    --outf HINER \
    --data_path /path/to/hsi.tif \
    --vid sample_spectral_sr \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --data_split 1_1_4 \
    --dec_strds 5 3 3 2 2 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --sr_ratio 4 \
    --modelsize 1.5 \
    -e 300 \
    --eval_freq 5 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001
```

</details>

### 3.5 Compression

First train a regression model and save the checkpoint. Then pass the checkpoint to `example/compress.py`.

<details>
<summary>Show command</summary>

```bash
python example/compress.py \
    --data_path /path/to/hsi.tif \
    --weight /path/to/model_best.pth \
    --quant_model_bit 8 \
    --quant_embed_bit 6 \
    --outf HINER \
    --vid sample_compression \
    --data_type HSI \
    --embed pe_1.25_80 \
    --crop_list 720_540 \
    --ori_shape 696_520_34 \
    --resize_list -1 \
    --arch hiner_modulate \
    --conv_type convnext pshuffel \
    --act gelu \
    --norm none \
    --loss Fusion10_freq \
    --enc_dim 64_16 \
    --fc_hw 4_3 \
    --reduce 1.2 \
    --dec_strds 5 3 3 2 2 \
    --ks 0_1_5 \
    --data_norm 4096 \
    --modelsize 0.5 \
    -e 300 \
    --eval_freq 30 \
    --lower_width 12 \
    -b 1 \
    --lr 0.001
```

</details>

## Important Arguments

| Argument | Description |
| --- | --- |
| `--data_path` | Path to `.mat` or `.tif` HSI data. |
| `--clean_data_path`, `--noise_data_path` | Clean/noisy pair for real denoising. |
| `--ori_shape` | Original HSI shape as `H_W_C`. |
| `--crop_list` | Spatial crop or padded size as `H_W`; use `-1` to disable. |
| `--resize_list` | Resize target; use `-1` to disable. |
| `--data_norm` | Divide input values by this constant; use `-1` for min-max normalization per band. |
| `--arch` | Use `hiner_modulate` for HINER++. |
| `--modelsize` | Approximate model capacity in millions of parameters. |
| `--restore` | Restoration mode for denoising, inpainting, or spatial SR. |
| `--sr_ratio` | Spectral super-resolution ratio. |
| `--dump_images` | Save visualized predictions during evaluation. |

## Outputs

By default, experiments write to:

```text
output/<outf>/<experiment_id>/
```

The folder contains checkpoints, logs, optional visualizations, and exported `.mat` reconstruction results.

## Citation

If this code is useful for your research, please cite:

```bibtex
@article{shi2024compression,
  title={Compression as Restoration: A Unified Implicit Approach to Self-Supervised Hyperspectral Image Representation},
  author={Shi, Junqi and Zhang, Qirui and Lu, Ming and Ma, Zhan},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2024},
  publisher={IEEE}
}
```

## Contact

- Junqi Shi: junqishi@smail.nju.edu.cn
- Qirui Zhang: qiruizhang@smail.nju.edu.cn
- Zhan Ma: mazhan@nju.edu.cn

## License

Please add a license file before publishing the repository. If you want others to freely use and modify the code, MIT or Apache-2.0 are common choices.
