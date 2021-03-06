# NDN4-tf2-pytorch-exploration
Exploration into replacing/reimplementing tools provided by [NeuroTheoryUMD/NDN3](https://github.com/NeuroTheoryUMD/NDN3) with/in TF2 and/or PyTorch. Starting to contain reimplementation of bunch of [msc-neuro](https://github.com/petrroll/msc-neuro/) models.

So far only a personal playground without _any_ goals. 

## Ideas to explore (no particular order): 
- [x] Laplacian2D regularization
- [x] Pearson's R metric 
- [x] Non-DoG models (rLN, rLNLN, conv) from msc-neuro
- [x] Rudimentary performance evaluation
  - Seems to be ~fast/slow for msc-neuro-like models
- [x] DoG layer
- [x] Reimplementation of baseline 4 from msc-neuro
  - [ ] Match loss computation 100 %
- [ ] NDN3's data filters
- [ ] Consider cleaning up tools ^^ and packaging them

## Setup: 
- Create & activate virtual environment `python3 -m venv env` (or conda, or ...)
- Install required packages `pip install -r ./requirements.txt`
- Hydrate `./Data`

### Data:
Scripts assume freely available data from [Model Constrained by Visual Hierarchy Improves Prediction of Neural Responses to Natural Scenes](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927) paper.
- Navigate to [Supporting Information](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927#sec018)
- Download the [first supplement](https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=info:doi/10.1371/journal.pcbi.1004927.s001)
- Unzip it to `./` (to have folder `./Data` in repo root with three subdirectories `region1` to `region3`)