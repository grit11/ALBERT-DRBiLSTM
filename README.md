# ALBERT-DRBiLSTM

Project code for fake review detection with `ALBERT + DRBiLSTM + AENSF`.

## Included Files

- `main.py`: training and evaluation entry
- `model/model_albert_bilstm.py`: main model
- `model/model_component.py`: DRBiLSTM and metadata attention components
- `requirements.txt`: project dependencies

## Data Format

Prepare three tab-separated files:

- `data/1.txt`: train set
- `data/3.txt`: dev set
- `data/2.txt`: test set

Each line must contain 6 columns:

```text
review<TAB>label<TAB>max_similarity<TAB>suspicion_score<TAB>timestamp<TAB>sentiment
```

Example:

```text
物流很快，包装很好	1	0.84	0.67	2023-05-11	5
```

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python main.py \
  --train-path data/1.txt \
  --dev-path data/3.txt \
  --test-path data/2.txt \
  --output-dir outputs \
  --pretrained-model-name albert-base-chinese
```

## Notes

- Data preprocessing scripts are not included in this repository snapshot.
- Outputs such as `best_model.pth`, `run_config.json`, and evaluation results will be saved to `outputs/`.
