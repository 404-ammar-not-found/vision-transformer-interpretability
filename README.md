# Vision Transformer Research Notes in PyTorch

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

This repository is an early-stage research notebook set for studying Vision Transformers in PyTorch. It combines a from-scratch implementation, fine-tuning experiments on Oxford Flowers 102, and attention visualisation work. The emphasis is on clear experimental notes, reproducible artefacts, and traceable source material rather than on packaging a production library.

## Research Aim

- Reconstruct the ViT pipeline from first principles.
- Fine-tune a pretrained ViT on Oxford Flowers 102.
- Inspect patch embeddings, class tokens, positional embeddings, QKV projections, and multi-head attention.
- Record model summaries, figures, and intermediate results as a laboratory-style log.

## Source Material

External sources used in the project:

1. Dosovitskiy et al., [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).
2. [Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset.
3. [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/stable/index.html), [timm](https://github.com/huggingface/pytorch-image-models), [torchinfo](https://github.com/TylerYep/torchinfo), and [matplotlib](https://matplotlib.org/).

Internal source material in this repository:

- [src/understanding_vit_from_scratch.ipynb](src/understanding_vit_from_scratch.ipynb)
- [src/finetuning_vit.ipynb](src/finetuning_vit.ipynb)
- [src/understanding_how_vits_see.ipynb](src/understanding_how_vits_see.ipynb)
- [summaries/vit_summary.txt](summaries/vit_summary.txt)
- [summaries/patch_embedding_summary.txt](summaries/patch_embedding_summary.txt)
- [summaries/transformer_encoder_summary.txt](summaries/transformer_encoder_summary.txt)
- [ViT_summary.txt](ViT_summary.txt)

## Repository Record

| Path | Role in the study |
| --- | --- |
| [src/understanding_vit_from_scratch.ipynb](src/understanding_vit_from_scratch.ipynb) | From-scratch ViT reconstruction, dataset handling, patching, and architecture notes. |
| [src/finetuning_vit.ipynb](src/finetuning_vit.ipynb) | Fine-tuning a pretrained ViT on Oxford Flowers 102. |
| [src/understanding_how_vits_see.ipynb](src/understanding_how_vits_see.ipynb) | Attention reconstruction and head-level visualisation. |
| [summaries/](summaries/) | Short technical summaries of the patch embedding, transformer encoder, and overall ViT design. |
| [figures/](figures/) | Saved plots, sample images, and attention maps. |
| [src/archive/flowers102/](src/archive/flowers102/) | Local dataset cache used by the notebooks. |

## Checkpoints And Artefacts

- `src/pretrained_models/vit_base_16.pth`: pretrained ViT weights used as a starting point.
- `src/finetuned_models/best_model.pth`: the finetuned Flowers 102 checkpoint.
- `figures/`: exported plots from the notebook runs.
- `summaries/`: concise written summaries of the model blocks and architecture.

## Experimental Setup

- Core model family: ViT base with patch size 16.
- Common image resolutions: 224 px in the from-scratch material and 384 px in the fine-tuning and attention notebooks.
- Primary classification task: Oxford Flowers 102.
- Visual outputs and checkpoints are written to the repository so the experiments can be inspected after execution.

## Reproducing The Notebooks

```bash
git clone https://github.com/404-ammar-not-found/ViT-In-Pytorch.git
cd ViT-In-Pytorch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebooks in `src/` and run them in this order if you want the full narrative:

1. `understanding_vit_from_scratch.ipynb`
2. `finetuning_vit.ipynb`
3. `understanding_how_vits_see.ipynb`

The notebooks will populate `figures/` and `src/finetuned_models/` with plots and checkpoints as the experiments run.

## Notes On The Record

- This is an exploratory research codebase, not a polished framework.
- The notebooks are intentionally verbose so that the reasoning is easy to audit later.
- If you are extending the work, keep new observations in the same source-and-artefact style so the record remains coherent.

## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE) for details.