This repository provides input files and analysis scripts for our computational study of human Hv1 (hHv1) dimerization.

- Initial PDB structures and corresponding AMBER topology files (`.top`) for the five candidate dimer models.
- Recorded MD random seeds and trajectory identifiers in [md_seeds.txt](md_seeds.txt).
- Scripts for MM/GBSA, normal-mode (NMODE) entropy calculations, and β-VAE analysis.
 It also records the six additional aqueous S1–S1′ trajectories.  The β-VAE training seeds were 11, 29, 47, 71, and 101.

The original MD trajectories and extracted analysis frames are omitted because of their size. To rerun the simulations, use the supplied structures and topology files, the recorded seeds, and the MD protocol described in the manuscript. Before running the analysis scripts, select frames from the resulting trajectories using the sampling procedure reported in the manuscript.
