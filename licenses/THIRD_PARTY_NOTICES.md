# Third-party material redistributed in this repository

The IMPaCT Synergy Pipeline code is released under the MIT License
(`LICENSE`, identical copy in `licenses/MIT_LICENSE`). The MIT License does
**not** cover the brain atlases stored under `atlases/`. They are third-party
data, redistributed unmodified under their own terms and listed below.
`scripts/download_atlases.sh --verify` checks the local copies against the
SHA-256 values in this table, and `scripts/download_atlases.sh` re-downloads
them from the pinned sources.

| Local file | Source (pinned) | SHA-256 | License / terms |
| --- | --- | --- | --- |
| `atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz` | ThomasYeoLab/CBIG, tag `v0.14.3-Update_Yeo2011_Schaefer2018_labelname`, `stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/` (the URL used by nilearn 0.13 `fetch_atlas_schaefer_2018`) | `abb8032840af30603995fd59634fd6bcd816088870de9061bbf90ad42c408d4d` | MIT (CBIG repository `LICENSE.md`) |
| `atlases/schaefer_2018/Schaefer2018_400Parcels_7Networks_order.txt` | same as above | `f62cdc9de9696e11bda9fe1cd51fc013b1da27ba180d2dcead8f3fd94bf3cbc1` | MIT (CBIG repository `LICENSE.md`) |
| `atlases/aal_SPM12/aal/` (the pipeline reads `atlas/AAL.nii`, 116 labels) | GIN Bordeaux, `https://www.gin.cnrs.fr/AAL_files/aal_for_SPM12.tar.gz` (AAL for SPM12, 2015-08-25; the URL used by nilearn `fetch_atlas_aal(version="SPM12")`) | `AAL.nii`: `91e8ec62a293c2a28c8e7343f37abb978ab941c5e26fbc496945c66d073a2cd8` | GNU General Public License, as stated in `atlases/aal_SPM12/aal/readme_aal_for_SPM12.txt` |
| `atlases/shen_1mm_268_parcellation.nii.gz` | canlab/Neuroimaging_Pattern_Masks, commit `788c2108df4145e4abb6218ebda08e406437006e`, `Atlases_and_parcellations/2013_Shen_Constable_NIMG_268_parcellation/` (redistribution of the Yale BioImage Suite / NITRC parcellation) | `675f9174c0c418e30f95e7e9fd84be4fdf3b705656fbb22164aeca8546f0c21d` | GPL-3.0 (canlab repository `LICENSE`) |

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

External tools that the helper scripts call are not redistributed here:
fMRIPrep (`nipreps/fmriprep`, Apache-2.0) runs as its own container, and
FreeSurfer requires a personal license file (see `fs_license.txt.example`),
which must never be committed.
