```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip3 install -r requirements.txt
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