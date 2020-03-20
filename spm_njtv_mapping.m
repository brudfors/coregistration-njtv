function M = spm_njtv_mapping(fix,mov,in,R,ix)
% Get transformation from fixed to moving, based on registration results
% from spm_njtv_coreg.
% FORMAT M = spm_njtv_mapping(fix,mov,in,R,ix)
%
% INPUT
% fix - Index of fixed image
% mov - Index of moving image
% in  - Images as Niis or filenames
% R   - Rigid transformation matrices
% ix  - struct with indices of fixed and moving images (see spm_njtv_coreg)
%
% OUTPUT
% M   - Transformation from mov to fix
% _______________________________________________________________________
%  Copyright (C) 2020 Wellcome Trust Centre for Neuroimaging