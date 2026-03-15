# InstaSHAP — PyTorch Reimplementation
A PyTorch reimplementation of InstaSHAP (Enouen & Laguna, ICLR 2025), developed as part of Research Seminar in Explainable AI: Reproducibility and Critical Evaluation of Methods, supervised by Dr. Giuseppe Casalicchio.

Note: This is an independent reimplementation for academic study and critical analysis, not the authors' official code.

## Repository Structure
 
```
InstaSHAP_implement/
├── src/
│   ├── instashap.py          # Core implementation
│   ├── dataloader.py          # Dataset loading & preprocessing
│   └── distill_utils.py       # Knowledge distillation & data augmentation
├── test/
│   ├── replication.py                   # Reproduce paper experiments (Fig 3 & 4)
│   ├── simulation.ipynb                 # Synthetic data experiments
│   ├── test_instaSHAP.ipynb             # Bike Sharing experiments
│   ├── test_instaSHAP_treeover.ipynb    # Tree Cover Type experiments
│   ├── data_augment_distillation.ipynb  # Knowledge distillation experiment
│   ├── hypermeter_tuing_bikesharing.py  # Hyperparameter tuning script
│   └── results/                         # Saved figures
├── data/
│   ├── bike_sharing/          # Bike Sharing dataset (hour.csv, day.csv)
│   └── covertype/             # Forest Cover Type dataset (covtype.data.gz)
├── others/
│   ├── sparse-interaction-additive-networks.zip   # Patched SIAN package
│   ├── InstaSHAP_flowchart.png
│   └── flowchart.drawio
├── LICENSE
└── README.md
```
## Setup
 
### 1. Clone the repository
 
```bash
git clone https://github.com/yuysky/InstaSHAP_implement.git
cd InstaSHAP_implement
```
 
### 2. Install Python dependencies
 
```bash
pip install torch numpy matplotlib tqdm xgboost scikit-learn
```
 
### 3. Install the patched SIAN package
 
InstaSHAP relies on [SIAN (Sparse Interaction Additive Networks)]([https://github.com/matteo-enovouen/sparse-interaction-additive-networks](https://github.com/EnouenJ/sparse-interaction-additive-networks.git)) for interaction detection. The official package has incorrect imports in some files, so we ship a patched version.
 
```bash
cd others/
unzip sparse-interaction-additive-networks.zip
cd sparse-interaction-additive-networks/
pip install .
cd ../..
```
## Run the experiments

### Reproducing paper experiments
 
The `test/replication.py` script reproduces the two main experiments:
 
```bash
cd test/
python replication.py
```
 
This runs:
- **Experiment 1** (Section 6.1, Figure 3): 10D synthetic function — compares InstaSHAP vs FastSHAP on ground-truth Shapley values.
- **Experiment 2** (Section 6.2, Figure 4): Bike Sharing interaction detection — evaluates recovered feature synergies.
 
Results are saved to `test/results/`.

### Hyperparameter tuning

The script `test/hypermeter_tuing_bikesharing.py` performs a grid search over GAM order (`k=1,2,3`) and number of interactions (`N`) on the Bike Sharing dataset. 

```bash
cd test/
python hypermeter_tuing_bikesharing.py
```
### Knowledge distillation experiment
 
The notebook `test/data_augment_distillation.ipynb` demonstrates using XGBoost as a teacher model with mixup-based data augmentation to improve InstaSHAP training. See `src/distill_utils.py` for the implementation.


@misc{enouen2025instashapinterpretableadditivemodels,
      title={InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly}, 
      author={James Enouen and Yan Liu},
      year={2025},
      eprint={2502.14177},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.14177}, 
}
