# hHv1-dimer-analysis
Analysis scripts for the human Hv1 (hHv1) dimerization study.

This repository provides input files and analysis scripts for our computational
study of human Hv1 (hHv1) dimerization.

- Initial PDB structures and corresponding topology files (`.top`) for the five
  candidate dimer models.
- Molecular dynamics (MD) restart files (`.rst`) and recorded random seeds for
  rerunning the simulations.
- Scripts for MM/GBSA calculations and β-VAE analysis.

The original MD trajectories and extracted analysis frames are omitted because
of their size. To rerun the simulations, use the supplied topology and restart
files, the recorded seeds, and the MD protocol described in the manuscript.
Select frames from the rerun trajectories using the sampling procedure reported
in the manuscript before running the analysis scripts.
