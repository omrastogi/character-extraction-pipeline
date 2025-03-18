```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip3 install -r requirements.txt
```


```bash
pip install dghs-imgutils
```

```bash
character-extraction-pipeline/
│── data/                        # Dataset storage
│── docs/                        # Documentation
│── models/                      # Trained models and configs
│── src/                         # Main source code
│   ├── preprocess.py            # Image pre-processing
│   ├── attribute_extraction.py   # Model-based attribute extraction
│   ├── inference.py             # Pipeline for running inference
│   ├── utils.py                 # Utility functions
│── tests/                       # Test scripts
│── notebooks/                   # Jupyter notebooks for debugging
│── requirements.txt             # Dependencies
│── README.md                    # Documentation
│── .gitignore                   # Ignore unnecessary files
│── environment.yml               # Conda environment file
```


### Doubts to ask 

1. Should I also consider the existing description?

## Testing

#### Pipeline 1: VLM

| Input Image   |   Accuracy |   Time (seconds) |
|:--------------|-----------:|-----------------:|
| 1             |          8 |          17.56   |
| 2             |          9 |          15.09   |
| 3             |          8 |          14.43   |
| 4             |         10 |          13.11   |
| 5             |          9 |          13.75   |
| 6             |         11 |           9.75   |
| 7             |          9 |          11.79   |
| 8             |         10 |          13.11   |
| 9             |         11 |           9.58   |
| 10            |         12 |          16.44   |
| 11            |          7 |          31.6    |
| 12            |         10 |          12.35   |
| 13            |          9 |          13.8    |
| 14            |         10 |          13.24   |
| 15            |          7 |          14.52   |
| Average       |          9 |          14.6747 |
-------------------------------------------------

#### Pipeline 2: Tagger + VLM

| Input Image   |   Accuracy |   Time (seconds) |
|:--------------|-----------:|-----------------:|
| 1             |       10   |         10.81    |
| 2             |       11   |          8.01    |
| 3             |       11   |          9.38    |
| 4             |        8   |          7.97    |
| 5             |       11   |          8.39    |
| 6             |       12   |          8.96    |
| 7             |       12   |         10.19    |
| 8             |       11   |          9.16    |
| 9             |        8   |          6.85    |
| 10            |       12   |         11.17    |
| 11            |       11   |         10.79    |
| 12            |       11   |          6.84    |
| 13            |       12   |          8.16    |
| 14            |       11   |          6.79    |
| 15            |       11   |          8.03    |
| Average       |       10.8 |          8.76667 |
-------------------------------------------------

