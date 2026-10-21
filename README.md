# Model Quantization

Quantize [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) using raw PyTorch. The notebook compares symmetric and asymmetric INT8 with per-tensor, per-channel, and per-group quantization.

**Goal:** reduce checkpoint size by at least 45% with no more than 1 percentage point of accuracy loss.

## Run

Use **Python 3.12**. From the project folder:

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m jupyter lab model_quantizer.ipynb
```

Run the notebook cells in order. Internet access is needed to download the model and dataset.

## Results

**Winner: asymmetric per-channel INT8.**

| Model | Checkpoint size | Validation accuracy |
| --- | ---: | ---: |
| Original FP32 | 255.45 MiB | 91.06% |
| Selected INT8 | 132.49 MiB | 91.06% |

The winner saved **48.13%** of storage and was the smallest configuration matching baseline accuracy on 872 validation examples. It was about **27% slower** in the CPU test because weights are reconstructed to FP32 for computation.

## Files

- [model_quantizer.ipynb](model_quantizer.ipynb): quantization and evaluation code.
- [quantizer.py](quantizer.py): quantizer and saved-model loader; keep it beside the notebook.
- [results/results.csv](results/results.csv): comparison of all configurations.
- [quantization_report.pdf](quantization_report.pdf): short report.

The notebook exports the winner to `artifacts/hf_model/`. Upload that folder to Hugging Face; its custom checkpoint requires the included loader to run predictions.
