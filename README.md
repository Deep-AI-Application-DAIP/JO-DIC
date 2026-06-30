# JO-DIC Reproduction Package

This repository provides a compact PyTorch reproduction framework for **JO-DIC: Binocular Full-Field 3D Displacement and Strain Measurement via Spatial–Temporal Joint Optimization**.

The project integrates the main algorithmic components described in the paper:

* a CGI-Stereo-like binocular disparity estimation branch;
* a RAFT-like temporal optical-flow estimation branch;
* Linear Deformable Convolution (LDConv) for adaptive local speckle feature enhancement;
* Manhattan Self-Attention (MaSA) for long-range spatial structural modeling;
* spatial–temporal geometric consistency loss for joint optimization;
* binocular triangulation, 3D displacement reconstruction, and strain-field calculation.

This package is intended as a runnable research reproduction framework. It does not include pretrained weights or the original industrial airship dataset.

---

## 1. Project Structure

The recommended project structure is:

```text
JO-DIC/
  configs/
    jodic_synthetic.json

  data/
    SyntheticSpeckleDataset/
      raw.zip
      raw/

  jodic/
    datasets/
      __init__.py
      synthetic_speckle.py

    models/
      __init__.py
      feature.py
      geometry.py
      jodic.py
      ldconv.py
      losses.py
      masa.py
      raft_like.py
      stereo_like.py
      strain.py

    __init__.py
    __main__.py
    demo_forward.py
    train.py

  tools/
    build_synthetic_dataset.py

  main.py
  README.md
  requirements.txt
```

The core runnable implementation is located in the `jodic/` directory. The `data/` directory is used for storing user-provided datasets.

---

## 2. Dataset Placement

The unprocessed Synthetic Speckle Dataset mentioned in the paper should be placed under:

```text
data/SyntheticSpeckleDataset/
```

If the dataset is provided as a compressed file, place it as:

```text
data/SyntheticSpeckleDataset/raw.zip
```

If the dataset is extracted, place the extracted files under:

```text
data/SyntheticSpeckleDataset/raw/
```

The current default training script does not directly read `raw.zip` or `raw/`. It uses the built-in synthetic speckle generator implemented in:

```text
jodic/datasets/synthetic_speckle.py
```

Therefore, the project can run even without an external dataset.

To train directly on the unprocessed Synthetic Speckle Dataset, a dedicated raw dataset reader should be added according to the actual internal file structure of the dataset.

---

## 3. Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

A compatible PyTorch version is required. If GPU acceleration is needed, install the PyTorch version matching your CUDA environment.

---

## 4. Quick Forward Test

Run a single forward pass:

```bash
python main.py demo
```

Equivalent package command:

```bash
python -m jodic demo
```

Expected output:

```text
JO-DIC forward test
     disp0: (1, 1, H, W)
     disp1: (1, 1, H, W)
      flow: (1, 2, H, W)
    disp3d: (1, 3, H, W)
strain:
   eps_xx: (1, 1, H, W)
   eps_yy: (1, 1, H, W)
   eps_xy: (1, 1, H, W)
   eps_zx: (1, 1, H, W)
   eps_zy: (1, 1, H, W)
   eps_zz: (1, 1, H, W)
Done.
```

The outputs correspond to:

```text
disp0  = disparity at reference time t0
disp1  = disparity at current time t1
flow   = optical flow from t0 to t1
disp3d = reconstructed 3D displacement field [U, V, W]
strain = reconstructed strain-field components
```

---

## 5. Training on Generated Synthetic Speckle Data

The current training script uses the built-in synthetic speckle generator. To start training, run:

```bash
python main.py train --epochs 2 --batch-size 2 --height 128 --width 160 --max-disp 48
```

Equivalent package command:

```bash
python -m jodic train --epochs 2 --batch-size 2 --height 128 --width 160 --max-disp 48
```

The default training objective is:

```text
L = lambda_d * L_disparity + lambda_f * L_flow + lambda_c * L_consistency
```

where:

```text
L_disparity   = supervised disparity regression loss
L_flow        = supervised optical-flow regression loss
L_consistency = spatial–temporal geometric consistency loss
```

The spatial–temporal consistency term is:

```text
L_c = rho( warp(d_t1, flow) - d_t0 )
```

This term enforces consistency between temporal optical flow and stereo disparity evolution.

---

## 6. Generating a Small Synthetic Dataset Example

To generate example synthetic speckle samples for inspection, run:

```bash
python main.py build-data --out synthetic_speckle_demo --n 16
```

This command creates a small folder containing image quadruples and corresponding labels.

If this functionality is not needed, the `tools/` directory can be removed.

---

## 7. Main Commands

Print project information:

```bash
python main.py info
```

Run a forward test:

```bash
python main.py demo
```

Train on generated synthetic speckle data:

```bash
python main.py train --epochs 2 --batch-size 2
```

Build a small synthetic speckle dataset example:

```bash
python main.py build-data --out synthetic_speckle_demo --n 16
```

Equivalent package commands:

```bash
python -m jodic info
python -m jodic demo
python -m jodic train --epochs 2 --batch-size 2
```

---

## 8. Module Correspondence

| Paper Component                                            | Implementation File                   |
| ---------------------------------------------------------- | ------------------------------------- |
| Adaptive Feature Enhancement / LDConv                      | `jodic/models/ldconv.py`              |
| Manhattan Self-Attention / MaSA                            | `jodic/models/masa.py`                |
| Stereo disparity estimation branch                         | `jodic/models/stereo_like.py`         |
| RAFT-like temporal optical-flow branch                     | `jodic/models/raft_like.py`           |
| Spatial–temporal geometric consistency loss                | `jodic/models/losses.py`              |
| Binocular triangulation and 3D displacement reconstruction | `jodic/models/geometry.py`            |
| Strain-field reconstruction                                | `jodic/models/strain.py`              |
| Complete JO-DIC model                                      | `jodic/models/jodic.py`               |
| Synthetic speckle data generator                           | `jodic/datasets/synthetic_speckle.py` |

---

## 9. Model Input and Output

The complete JO-DIC model receives four input images:

```python
left0   # left camera image at reference time t0
right0  # right camera image at reference time t0
left1   # left camera image at current time t1
right1  # right camera image at current time t1
```

Example:

```python
import torch
from jodic.models.jodic import JODICModel

model = JODICModel(max_disp=64)

left0 = torch.rand(1, 1, 256, 512)
right0 = torch.rand(1, 1, 256, 512)
left1 = torch.rand(1, 1, 256, 512)
right1 = torch.rand(1, 1, 256, 512)

intrinsics = {
    "fx": 1200.0,
    "fy": 1200.0,
    "cx": 256.0,
    "cy": 128.0,
    "baseline": 80.0
}

outputs = model(left0, right0, left1, right1, intrinsics=intrinsics)
```

Main outputs:

```python
outputs["disp0"]    # disparity at t0
outputs["disp1"]    # disparity at t1
outputs["flow"]     # optical flow from t0 to t1
outputs["disp3d"]   # 3D displacement field [U, V, W]
outputs["strain"]   # strain-field components
```

---

## 10. Calibration File Example

A recommended calibration file is:

```text
data/SyntheticSpeckleDataset/raw/calib.json
```

Example content:

```json
{
  "fx": 1200.0,
  "fy": 1200.0,
  "cx": 512.0,
  "cy": 256.0,
  "baseline": 80.0
}
```

The calibration parameters are used for binocular triangulation and 3D displacement reconstruction.

---

## 11. Cleaning Optional Runtime Files

The following files and directories are not required for normal execution and can be safely removed:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
```

On Windows PowerShell:

```powershell
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force
```

The following directories are optional:

```text
tools/
configs/
```

Keep `tools/` if you want to use:

```bash
python main.py build-data
```

Keep `configs/` if you want to preserve experiment configuration files.

The following files and directories should be kept for normal operation:

```text
main.py
jodic/
requirements.txt
README.md
data/
```

---

## 12. Notes

This repository provides a runnable reproduction framework for the JO-DIC pipeline. The numerical results reported in the paper require the original training data, validation data, camera calibration parameters, preprocessing details, training schedule, and pretrained weights.

The built-in synthetic speckle generator is intended for pipeline verification and debugging. For full reproduction using the paper's Synthetic Speckle Dataset, the raw dataset reader should be adapted to the exact dataset format.
