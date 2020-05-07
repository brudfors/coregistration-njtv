# NJTV: Groupwise Multimodal Image Registration using Joint Total  Variation

MATLAB implementation of an algorithm for groupwise, between modality coregistration using a Normalised Joint Total Variation (NJTV) cost function:

```
Brudfors M, Balbastre Y, Ashburner J. 
Groupwise Multimodal Image Registration using Joint Total  Variation.
arXiv preprint arXiv:2005.02933. 2020 May 7.
```

The algorithm aligns multiple images **rigidly**, by optimising the NJTV multi-channel cost function using Powell's method.

To run, simply give a bunch of nifti files (as char array or SPM nifti object) as input to the `spm_coregistration_njtv` function. This function will then output the estimated rigid transformation matrices (or an option can be set to change the orientation matrices in the nifti headers).

## Basic use case


```
% Paths to nifti images (*.nii)
pth_t1w = 'mri.nii'; % T1w MR image
pth_t2w = 'mri.nii'; % T2w MR image
pth_ct  = 'ct.nii';  % CT image
pth_in  = char(pth_t1w, pth_t2w, pth_ct); % Input to NJTV

% NJTV options
opt = struct('IxFixed', 1, ...          % T1w MRI is fixed
             'ShowAlign', 1, ...        % Enable some verbose
             'ShowFit4Scaling', true);  % Enable some more verbose

% Run NJTV registration
[~, R] = spm_njtv_coreg(pth_in, opt);  % More detailed instructions in this function
```

## Dependencies

The code requires that SPM12 and SPM's auxiliary-functions toolbox is on the MATLAB path:

* SPM12: Download from https://www.fil.ion.ucl.ac.uk/spm/software/download

* auxiliary-functions: git clone from https://github.com/WTCN-computational-anatomy-group/auxiliary-functions
