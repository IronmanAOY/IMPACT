# Third-party material redistributed in this repository

The IMPaCT Synergy Pipeline code is released under the MIT License
(`LICENSE`, identical copy in `licenses/MIT_LICENSE`). The MIT License does
**not** cover the brain atlases stored under `atlases/` or the structural
connectome under `data/managed/structural/`. They are third-party data,
redistributed under their own terms and listed below.

## Brain atlases

The atlases are redistributed unmodified.
`scripts/download_atlases.sh --verify` checks the local copies against the
SHA-256 values in this table, and `scripts/download_atlases.sh` re-downloads
them from the pinned sources.

| Local file | Source (pinned) | SHA-256 | License / terms |
| --- | --- | --- | --- |
| `atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz` | ThomasYeoLab/CBIG, tag `v0.14.3-Update_Yeo2011_Schaefer2018_labelname`, `stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/` (the URL used by nilearn 0.13 `fetch_atlas_schaefer_2018`) | `abb8032840af30603995fd59634fd6bcd816088870de9061bbf90ad42c408d4d` | MIT (CBIG repository `LICENSE.md`) |
| `atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt` | same as above | `f62cdc9de9696e11bda9fe1cd51fc013b1da27ba180d2dcead8f3fd94bf3cbc1` | MIT (CBIG repository `LICENSE.md`) |
| `atlases/aal_SPM12/aal/` (the pipeline reads `atlas/AAL.nii`, 116 labels) | GIN Bordeaux, `https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz` (AAL for SPM12, 2015-08-25; the URL used by nilearn `fetch_atlas_aal(version="SPM12")`) | `AAL.nii`: `91e8ec62a293c2a28c8e7343f37abb978ab941c5e26fbc496945c66d073a2cd8`; `AAL.xml`: `1a0f2bb952b700fa94f5156b544cf238f2e8d0e0e5ebedaefac835938b7609f3` | GNU General Public License, as stated in `atlases/aal_SPM12/aal/readme_aal_for_SPM12.txt` |
| `atlases/shen_1mm_268_parcellation.nii.gz` | canlab/Neuroimaging_Pattern_Masks, commit `788c2108df4145e4abb6218ebda08e406437006e`, `Atlases_and_parcellations/2013_Shen_Constable_NIMG_268_parcellation/` (redistribution of the Yale BioImage Suite / NITRC parcellation) | `675f9174c0c418e30f95e7e9fd84be4fdf3b705656fbb22164aeca8546f0c21d` | GPL-3.0 (canlab repository `LICENSE`) |

Of the AAL folder the pipeline reads only `atlas/AAL.nii`;
`scripts/download_atlases.sh` also verifies the label file `atlas/AAL.xml`.

Citations:

- Schaefer A, et al. Local-global parcellation of the human cerebral cortex
  from intrinsic functional connectivity MRI. Cereb Cortex 2018;28:3095-3114.
- Tzourio-Mazoyer N, et al. Automated anatomical labeling of activations in
  SPM using a macroscopic anatomical parcellation of the MNI MRI single-subject
  brain. NeuroImage 2002;15:273-289.
- Shen X, Tokoglu F, Papademetris X, Constable RT. Groupwise whole-brain
  parcellation from resting-state fMRI data for network node identification.
  NeuroImage 2013;82:403-415.

Note on the Shen file: the canlab source notes that this parcellation appears
to be defined in Colin27 space rather than MNI152 (see the `README.md` next to
the file in the canlab repository). Keep this in mind when it is applied to
fMRIPrep outputs in `MNI152NLin2009cAsym` space.

## Structural connectome

The MPC-Bench whole-brain model (`src/impact_pipeline/bench/whole_brain.py`)
reads this file and refuses a copy whose SHA-256 differs from the one below.

| Local file | Source | SHA-256 | License / terms |
| --- | --- | --- | --- |
| `data/managed/structural/budapest_connectome_3.0_209_0_median.csv` | Budapest Reference Connectome Server v3.0 (`https://pitgroup.org/connectome`, formerly `connectome.pitgroup.org`), consensus connectome computed from Human Connectome Project diffusion MRI. CSV edge-list export: 1000 edges, edge weight = median number of fibres, every edge with an edge confidence of at least 209; the file name records the export settings (version 3.0, 209, 0, median) | `40645a340981a30e5de3c2cddd4220b712e21396be865fd584f2a7920a9dd2b9` | No license is stated by the server; its terms of use ask users who publish results to cite the two publications below and the server address |

Citations:

- Szalkai B, Kerepesi C, Varga B, Grolmusz V. The Budapest Reference
  Connectome Server v2.0. Neurosci Lett 2015;595:60-62.
- Szalkai B, Kerepesi C, Varga B, Grolmusz V. Parameterizable consensus
  connectomes from the Human Connectome Project: the Budapest Reference
  Connectome Server v3.0. Cogn Neurodyn 2017;11:113-116.

## External tools

External tools that the helper scripts call are not redistributed here:
fMRIPrep (`nipreps/fmriprep`, Apache-2.0) runs as its own container, and
FreeSurfer requires a personal license file (see `fs_license.txt.example`),
which must never be committed.
