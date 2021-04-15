# NDN4-tf2-pytorch-exploration
Exploration into replacing/reimplementing tools provided by [NeuroTheoryUMD/NDN3](https://github.com/NeuroTheoryUMD/NDN3) with/in TF2 and/or PyTorch.

So far only a personal playground without _any_ goals.

# Setup: 
- Create & activate virtual environment `python3 -m venv env` (or conda, or ...)
- Install required packages `pip install -r ./requirements.txt`
- Hydrate `./Data`

## Data:
Scripts assume freely available data from [Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927) paper.
- Navigate to [Supporting Information](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927#sec018)
- Download the [first supplement](https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=info:doi/10.1371/journal.pcbi.1004927.s001)
- Unzip it to `./` (to have folder `./Data` in repo root with three subdirectories `region1` to `region3`)