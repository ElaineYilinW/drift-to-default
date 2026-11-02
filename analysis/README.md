# Analysis

Each preregistered study has one analysis script, and each is frozen by the SHA-256 hash in its preregistration. Do not edit frozen files; `verify_preregistration.py` checks them. Some docstrings cite the paths the preregistrations had when they were frozen; those documents are now in `docs/preregistrations/`.

| Files | Purpose |
|---|---|
| `verify_preregistration.py` | Compare the repository with the preregistered hashes |
| `analyze_sNN.py` | Analysis of study NN |
| `analyze_<topic>.py` | Analyses of the earlier studies, named by topic (index: `results/README.md`) |
| `make_*_tasks.py` | Task lists with the exact generation commands of a study |
| `analyze_sinkprobe*.py`, `analyze_sinkband.py` | Attention-to-sink probe and band probe (exploratory) |
| `robustness_*.py`, `encoder_robustness.py`, `supplementary_analyses*.py`, `motion_*.py`, `*_supp.py` | Robustness and supplementary analyses reported in the paper (exploratory) |
| `ar_equilibrium.py`, `analyze_relax_theory.py` | Linear chunk-recursion model of relaxation (exploratory) |
| `select_openings.py`, `s27_distances.py` | Openings and distances chosen by fixed rules |
