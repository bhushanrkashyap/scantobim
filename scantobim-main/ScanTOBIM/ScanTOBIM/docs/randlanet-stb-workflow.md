# RandLA-Net Integration with STB (VS Code)

This workflow keeps STB unchanged and plugs RandLA-Net outputs into STB semantic masking.

## 1) Create a separate RandLA-Net environment

Use a dedicated environment so STB dependencies remain stable.

```powershell
python -m venv .venv-randla
.\.venv-randla\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy open3d
```

If your network policy blocks direct GitHub/PyPI installs, install the RandLA-Net repo manually from an approved internal mirror.

## 2) Export STB semantic-stage points

This exports the exact points STB uses before semantic masking.

```powershell
.\.venv\Scripts\Activate.ps1
python -m agent.tools.export_semantic_points --input "data\scan.e57" --output "artifacts\semantic_points.npy" --voxel-size 0.01
```

## 3) Run RandLA-Net inference on exported points

Run inference in `.venv-randla` and produce one value per point in the same order:

- boolean mask, or
- class ids, or
- probabilities

Save to `artifacts\randla_predictions.npy` (or `.npz`).

## 4) Build an STB checkpoint file

Convert model output to an STB-compatible wall mask and validate point count.

```powershell
.\.venv\Scripts\Activate.ps1
python -m agent.tools.build_semantic_checkpoint --points "artifacts\semantic_points.npy" --predictions "artifacts\randla_predictions.npy" --output "artifacts\wall_mask.npy" --mode auto --wall-class-id 1 --threshold 0.50
```

## 5) Run STB with semantic mode enabled

```powershell
.\.venv\Scripts\Activate.ps1
python -m agent.tools.processor_cli --input "data\scan.e57" --output "artifacts\results.json" --semantic-enable --semantic-model randlanet --semantic-checkpoint "artifacts\wall_mask.npy" --semantic-wall-threshold 0.50
```

Do not pass `--semantic-no-fallback` until quality is validated.

## 6) Validate in Revit

Compare geometry-only vs semantic-assisted runs for:

- wall completeness
- false positives
- family/component placement quality

Repeat with tuning of RandLA-Net threshold/class mapping.
