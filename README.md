# Regularized ERM on Random Subspaces

Code accompanying the paper:

**Regularized ERM on Random Subspaces**  
*A. Della Vecchia, J. Mourtada, E. De Vito, L. Rosasco*  
International Conference on Artificial Intelligence and Statistics (AISTATS), 2021

📄 [Paper link](https://proceedings.mlr.press/v130/della-vecchia21a.html)  
🔗 [Google Scholar](https://scholar.google.com/citations?user=aaeUheEAAAAJ)

---

## Overview
This repository contains simulation code and experiments for studying kernel-based regularized empirical risk minimization (ERM) on random subspaces, with a particular emphasis on the Nyström approximation.

Our work investigates the statistical–computational trade-offs that arise when applying convex (and possibly non-differentiable) loss functions, such as the hinge loss for classification. The methodology is designed for scalability in high-dimensional settings, delivering significant memory and runtime savings while maintaining optimal performance.

## Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/andreadvecchia/svm-random-subspaces.git
cd svm-random-subspaces
pip install -r requirements.txt
```

---

## Usage
You can reproduce the experiments from the paper with:

```bash
python scripts/run_experiments.py --dataset synthetic --loss hinge --subspace nystrom
```

To plot the results:
```bash
python scripts/plot_results.py --input results/exp1.csv
```

(We will update the available options once the scripts are fully cleaned.)

---

## Repository Structure
```
svm-random-subspaces/
│
├── src/                # core algorithms and helper functions
├── scripts/            # run experiments and plotting
├── data/               # data or generators
├── results/            # experimental outputs and figures
├── notebooks/          # demo notebooks (optional)
├── requirements.txt    # dependencies
└── README.md           # project description
```

---

## Citation
If you use this code, please cite our paper:

```bibtex
@inproceedings{della2021regularized,
  title={Regularized ERM on random subspaces},
  author={Della Vecchia, Andrea and Mourtada, Jaouad and De Vito, Ernesto and Rosasco, Lorenzo},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4006--4014},
  year={2021},
  organization={PMLR}
}
```

---

## License
This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.


## Quickstart (Demo Notebook)

Open **`notebooks/demo_experiment.ipynb`** and run all cells to see Random‑Subspace SVM vs. a baseline SVM on a synthetic high‑dimensional dataset. The demo is self‑contained and finishes in a few seconds.

## Python API

```python

from svm_random_subspaces import RandomSubspaceSVM

model = RandomSubspaceSVM(M=25, k=20, random_state=7)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

```

