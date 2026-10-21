# Build your own model quantizer

Raw-PyTorch weight-only INT8 quantization of [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english). The notebook evaluates symmetric and asymmetric quantization with per-tensor, per-channel, and per-group scaling. It uses ordinary tensor operations rather than pre-built quantization libraries.

The constraint is at least **45% smaller checkpoint** with at most **1.0 percentage point of validation accuracy loss**. The recorded winner is **asymmetric per-channel INT8**: **132.49 MiB**, **48.13% smaller** than FP32, with **91.06% accuracy** on all 872 SST-2 validation examples. Computation remains FP32; the winner was about 27% slower in the recorded CPU timing.

## Install and run

Use **Python 3.10–3.12**; the recorded experiment used Python 3.12 and PyTorch 2.5.1 CPU. The pinned PyTorch and pandas releases do not support Python 3.14. From this project folder, create an environment separate from any existing `.venv`:

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m jupyter lab model_quantizer.ipynb
```

The first install command selects the CPU PyTorch build and avoids downloading CUDA packages. Its version also satisfies the `torch==2.5.1` requirement. On Windows, activate with `.venv-py312\Scripts\activate` instead of `source`.

Select this environment's Python kernel in your notebook editor. Run the cells from top to bottom. The first model and dataset downloads require internet access. Keep `quantizer.py` beside the notebook: the export cell copies it into the model folder, and the reload cell imports it.

## What the dependencies do

| Dependency | Purpose |
| --- | --- |
| `torch` | Quantization tensor operations and model inference |
| `transformers` | Original model architecture and tokenizer |
| `datasets` | SST-2 validation examples |
| `pandas` | Results table and CSV export |
| `nbformat`, `jupyterlab` | Notebook support and editor; JupyterLab also installs the Python kernel |

Transformers installs the compatible `huggingface_hub` dependency, which can also upload model files. The current notebook does not generate the PDF, so it does not require `reportlab`.

## Files and deliverables

- [model_quantizer.ipynb](model_quantizer.ipynb): quantizer, evaluation, winner selection, export, and reload check.
- [quantizer.py](quantizer.py): reusable quantizer and custom checkpoint loader.
- [results/results.csv](results/results.csv) and [results/measurements.json](results/measurements.json): recorded comparisons and experiment settings.
- [quantization_report.pdf](quantization_report.pdf): report included for submission.

Running the notebook creates `artifacts/hf_model/` with the winning checkpoint, model configuration, tokenizer, loader, and results. Checkpoints are excluded from GitHub by `.gitignore`; publish the winning model folder to a public Hugging Face model repository. Its custom format requires `load_quantized` from the included `quantizer.py` to run predictions.

The assignment deliverables are the public GitHub URL, public Hugging Face model URL, and reviewed 1–2 page PDF with the results table.
