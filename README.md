# DynaMesh4D

DynaMesh4D reconstructs temporally consistent human meshes from video. The pipeline uses SAM3 for prompted video masks, Foundation HMR (SAM-3D-Body) for per-frame MHR prediction, and a bidirectional Mamba smoother over the 266-dimensional pose sequence. The smoothed MHR parameters are decoded back to a mesh for every frame.

This release also includes an MHR-to-SMPL adapter. The main pipeline and the converter intentionally use two separate Conda environments because their dependency stacks are incompatible.

## Validated status

The release path was validated on a real 16-frame video with:

- Mamba input/output: `[1, 16, 266]`
- MHR mesh output: `[16, 18439, 3]`
- zero NaN/Inf values
- MHR-to-SMPL output: 6,890 vertices and 13,776 faces

**Use the included float32 compatibility configuration.** The upstream default SAM-3D-Body configuration enables BF16. In the validated integration, SAM3 leaves a global BF16 autocast active, and the MHR sparse CUDA operation fails because `addmm_sparse_cuda` does not support BFloat16. The default BF16 profile is therefore not supported by this release.

## Repository layout

- `newapp.py`: Gradio entry point and the SAM3 -> Foundation HMR -> Bi-Mamba pipeline
- `model_mamba.py`: bidirectional Mamba smoother
- `mamba_smoother_epoch_15.pth`: formal DynaMesh4D smoother weight
- `configs/inference_float32.yaml`: validated 16-frame-compatible main configuration
- `configs/sam3d_float32.yaml`: validated SAM-3D-Body float32 model configuration
- `mhr2smpl_adapter.py`: cross-environment MHR-to-SMPL adapter
- `mhr_worker.py`: `mhr_clean` worker entry point
- `third_party/MHR-main/`: the converter's required MHR source subset and mapping assets

## Environments

The exact environments are not merged and this repository intentionally does not provide an unverified lock file.

### `sam4d` (main pipeline)

Validated versions:

- Python 3.10.19
- PyTorch 2.5.0+cu121; torchvision 0.20.0+cu121
- CUDA 12.1
- mamba-ssm 2.3.0
- NumPy 2.2.6
- OmegaConf 2.3.0
- Gradio 6.0.2
- OpenCV 4.12.0.88
- trimesh 4.10.1

### `mhr_clean` (MHR-to-SMPL worker)

Validated versions:

- Python 3.12.13
- PyTorch 2.7.1+cpu
- pymomentum 0.1.0
- smplx 0.1.28
- NumPy 2.4.6
- SciPy 1.17.1
- trimesh 4.12.2

The validated converter worker runs on CPU because its pymomentum backend is CPU-only.

## External resources

Large foundation weights and licensed body-model assets are not included. Arrange the checkpoint root as follows:

```text
checkpoints/
├── sam3/sam3.pt
├── sam-3d-body-dinov3/
│   ├── model.ckpt
│   ├── model_config.yaml
│   └── assets/mhr_model.pt
├── moge-2-vitl-normal/model.pt
├── diffusion-vas-amodal-segmentation/
├── diffusion-vas-content-completion/
└── depth_anything_v2_vitl.pth
```

The paths above match `configs/inference_float32.yaml`. The existing project configuration records these upstream sources:

- SAM3: <https://huggingface.co/facebook/sam3>
- SAM-3D-Body: <https://huggingface.co/facebook/sam-3d-body-dinov3>
- MoGe-2: <https://huggingface.co/Ruicheng/moge-2-vitl-normal>
- Diffusion-VAS checkpoints: <https://github.com/Kaihua-Chen/diffusion-vas?tab=readme-ov-file#download-checkpoints>
- Depth Anything V2 Large: <https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true>

The converter additionally requires external MHR LOD1 assets and an SMPL neutral model file. Obtain them under their upstream access and redistribution terms, then set their paths as shown below. They are runtime resources, not Python source dependencies.

## Main-model quick start (recommended float32 path)

```bash
conda activate sam4d
cd /path/to/DynaMesh4D

export DYNAMESH4D_CONFIG="$PWD/configs/inference_float32.yaml"
export DYNAMESH4D_SAM3D_MODEL_CONFIG="$PWD/configs/sam3d_float32.yaml"
export DYNAMESH4D_CHECKPOINT_ROOT=/path/to/checkpoints
export DYNAMESH4D_OUTPUT_DIR=/path/to/outputs

python -B newapp.py
```

`newapp.py` launches the Gradio application on `0.0.0.0` with its existing sharing behavior. Select or upload a video, add a positive point on the person, run **Mask Generation**, and then run **4D Generation (Save Mesh)**.

The formal smoother weight is loaded from `mamba_smoother_epoch_15.pth` in the repository root. To place it elsewhere, set `DYNAMESH4D_MAMBA_CHECKPOINT` to the weight path.

## MHR-to-SMPL conversion

Run the adapter from `sam4d`; it starts the conversion worker in `mhr_clean`:

```bash
conda activate sam4d
cd /path/to/DynaMesh4D

export DYNAMESH4D_MHR_CLEAN_PYTHON=/path/to/miniconda3/envs/mhr_clean/bin/python
export DYNAMESH4D_MHR_ASSETS_ROOT=/path/to/MHR-main/assets
export SMPL_MODEL_PATH=/path/to/SMPL_NEUTRAL.pkl

python -B - <<'PY'
import os
import pickle
from mhr2smpl_adapter import MHRtoSMPLAdapter

with open("mhr_outputs.pkl", "rb") as handle:
    mhr_outputs = pickle.load(handle)

converter = MHRtoSMPLAdapter(os.environ["SMPL_MODEL_PATH"], device="cpu")
smpl_outputs = converter.process_video_outputs(mhr_outputs)

with open("smpl_outputs.pkl", "wb") as handle:
    pickle.dump(smpl_outputs, handle)
PY
```

For direct worker use inside `mhr_clean`, the verified CLI shape is:

```bash
conda activate mhr_clean
python -B mhr_worker.py \
  --input /path/to/mhr_input.pkl \
  --output /path/to/smpl_output.pkl \
  --smpl_model_path /path/to/SMPL_NEUTRAL.pkl \
  --device cpu \
  --mhr_assets_path /path/to/MHR-main/assets
```

## Optional smoother training

Training is not required for inference. If reproducing smoother training, keep AMASS outside the repository and point the script to the desired AMASS subset:

```bash
conda activate sam4d
export DYNAMESH4D_AMASS_DATASET=/path/to/AMASS/ACCAD
python -B train_mamba.py
```

This path setting only selects external training data; it does not change the training algorithm.

## Third-party notices

This repository contains selected third-party source needed by the validated runtime. Existing copyright and license notices have not been removed or rewritten. Before public redistribution, verify the applicable terms for every bundled third-party component and for all separately obtained weights and body-model assets.
