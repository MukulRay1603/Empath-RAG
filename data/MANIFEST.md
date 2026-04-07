# Data Manifest
Files tracked here are too large for git. Re-download using instructions below.

## data/raw/ — source datasets (not in git)
| Dataset | Source | Size | Download command |
|---|---|---|---|
| GoEmotions | HuggingFace | ~200MB | python src/data/download_datasets.py |
| Empathetic Dialogues | HuggingFace | ~500MB | python src/data/download_datasets.py |
| Reddit Mental Health | zenodo.org/records/3941387 | 3.1GB | zenodo_get 3941387 -o data/raw/reddit_mental_health/ |
| Suicide Detection | kaggle nikhileswarkomati/suicide-watch | 60MB | kaggle datasets download -d nikhileswarkomati/suicide-watch -p data/raw/suicide_detection/ --unzip |

## data/indexes/ — built artifacts (not in git)
| File | How to rebuild |
|---|---|
| faiss_flat.index | Run 
otebooks/colab_build_faiss_index.ipynb on Colab A100 |
| metadata.db | Same as above |

## data/processed/ — derived files (not in git)
| File | How to rebuild |
|---|---|
| nli_train/val/test.csv | python src/data/build_nli_pairs.py |
