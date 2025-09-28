# Regularized ERM on Random Subspaces

Code accompanying the paper:

**Regularized ERM on Random Subspaces**  
*Andrea Della Vecchia, Joseph Mourtada, Ernesto De Vito, Lorenzo Rosasco*  
International Conference on Artificial Intelligence and Statistics (AISTATS), 2021

📄 [Paper link](https://proceedings.mlr.press/v130/della-vecchia21a.html)  
🔗 [Google Scholar](https://scholar.google.com/citations?user=aaeUheEAAAAJ)

---

## Overview
This repository provides simulation code and experiments for studying **regularized empirical risk minimization (ERM)** on random subspaces, with applications to kernel methods such as the **Nyström method**.  
The work analyzes **statistical–computational tradeoffs** for convex loss functions, with a focus on scalability in high-dimensional settings.

---

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
@inproceedings{dellavecchia2021randomsubspaces,
  title={Regularized ERM on Random Subspaces},
  author={Della Vecchia, Andrea and Mourtada, Joseph and De Vito, Ernesto and Rosasco, Lorenzo},
  booktitle={Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2021}
}
```

---

## License
This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.
