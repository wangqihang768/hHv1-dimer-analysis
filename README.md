# hHv1 Dimerization: Simulation Inputs & Analysis Pipeline

This repository hosts the simulation input files, structural models, and analysis scripts supporting our computational investigation of human Hv1 (hHv1) dimerization.

## Repository Contents
* **Initial Models & Topologies:** Starting PDB structures and corresponding AMBER topology files (`.top`) for the five candidate hHv1 dimer configurations.
* **MD Seeds & Trajectory Logs:** Recorded molecular dynamics random seeds and trajectory identifiers in `md_seeds.txt`, which also documents the six additional aqueous S1–S1′ validation trajectories.
* **Free Energy & Machine Learning Scripts:** Scripts for MM/GBSA calculations, normal-mode (`NMODE`) vibrational entropy estimation, and $\beta$-VAE latent space modeling (using training seeds `11`, `29`, `47`, `71`, and `101`).

## Trajectory Frames & Planned Updates
Due to file size limitations, full-length raw MD trajectories are not hosted directly in this repository. To facilitate reproducibility and downstream validation, **this repository is under active maintenance and will be continuously updated with:**
1. **Conformational Ensembles:** Extracted analysis frames and key representative conformations along the dimerization pathways.
2. **Membrane-Protein Systems:** Equilibrated lipid bilayer–embedded hHv1 coordinates and associated system build files.
3. **Extended Analysis Suite:** Additional scripts for structural characterization, inter-helical contact mapping, and trajectory post-processing.

## Usage Notes
To rerun the simulations independently, pair the provided structures and `.top` files with the random seeds in `md_seeds.txt` following the simulation protocol reported in the manuscript. Prior to executing the analysis scripts, sample the trajectory frames according to the intervals and criteria specified in the Methods section of the manuscript.
