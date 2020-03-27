# Groupwise, rigid medical image alignment

MATLAB implementation of an algorithm for groupwise, between modality coregistration using a Normalised Joint Total Variation (NJTV) cost function. The algorithm aligns multiple images rigidly, by optimising a single multi-channel cost function. The idea is that the more images there are, the better the alignment. This could be useful, for example, in aligning Multi-Parameter Mapping (MPM) data, used to create quantitative Magnetic Resonance Images (qMRI). To run, simply give a bunch of nifti files (as char array or SPM nifti object) as input to the `spm_coregistration_njtv` function. This function will then output the estimated rigid transformation matrices.

The code requires that SPM12 and SPM's auxiliary-functions toolbox is on the MATLAB path:

* SPM12: Download from https://www.fil.ion.ucl.ac.uk/spm/software/download

* auxiliary-functions: git clone from https://github.com/WTCN-computational-anatomy-group/auxiliary-functions

That's it!
