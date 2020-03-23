function [q,res] = spm_njtv_coreg(varargin)
% Groupwise, between modality coregistration using a normalised joint 
% total variation (NJTV) cost function.
% FORMAT [q,res] = spm_njtv_coreg(in,opt)
%
% INPUT
% Nii - Images as Niis or filenames
% opt - Algorithm options (see below)
%
% OUTPUT
% q   - Lie algebra rigid parameterisation
% res - struct with ridig transformation matrices, and indices of fixed 
%       and moving images
% _______________________________________________________________________
%  Copyright (C) 2020 Wellcome Trust Centre for Neuroimaging

if nargin > 2
    % Step into Powell optimiser
    q = CostFun(varargin{:});
    return;
end

% Check if required toolboxes are on the MATLAB path
%-----------------------

% SPM12
% Download from https://www.fil.ion.ucl.ac.uk/spm/software/download
if isempty(fileparts(which('spm'))), error('SPM12 not on the MATLAB path (get at fil.ion.ucl.ac.uk/spm/software/download)'); end

% SPM auxiliary-functions
% git clone https://github.com/WTCN-computational-anatomy-group/auxiliary-functions
if isempty(fileparts(which('spm_gmm'))), error('SPM auxiliary-functions not on the MATLAB path (get at github.com/WTCN-computational-anatomy-group/auxiliary-functions)'); end

% Parse options
%-----------------------

if nargin == 1, opt = struct; end
% Convergence criteria [tx,ty,tz,rx,ry,rz]
if ~isfield(opt,'Tolerance'), opt.Tolerance = [0.02 0.02 0.02 0.001 0.001 0.001]; end
% Picks the fixed image [1]
if ~isfield(opt,'IdxFixed'), opt.IdxFixed = 1; end
% Show GMM/RMM fit to intensity histogram [false]
if ~isfield(opt,'ShowFit4Scaling'), opt.ShowFit4Scaling = false; end
% Show alignment: 0. nothing,1. each coarse-to-fine step, 2. live [1]
if ~isfield(opt,'ShowAlign'), opt.ShowAlign = 1; end
% Coarse-to-fine sampling scheme in decreasing order [8 4 2 1]
if ~isfield(opt,'Samp'), opt.Samp = [8 4 2 1];  end
tol        = opt.Tolerance;
ixf        = opt.IdxFixed;
show_fit   = opt.ShowFit4Scaling;
show_align = opt.ShowAlign;
samp       = opt.Samp;

% If SPM has been compiled with OpenMP, this will speed things up
setenv('SPM_NUM_THREADS',sprintf('%d',-1));

% Read image data
%-----------------------

Nii = varargin{1};
if ~isa(Nii,'nifti'), Nii = nifti(Nii); end

% Parameters
is2d = numel(Nii(1).dat.dim) == 2;
C    = numel(Nii); % Number of channels
chn  = 1:C;
ixm  = chn(~ismember(chn,ixf));
Nm   = numel(ixm); % Number of moving images
nq   = 6;          % Number of transformation parameters (per image)
if is2d
    % Input images are 2D
    nq  = 3;
    tol = tol([1 2 6]); % selects x and y translation, and rotation component    
end
sc0  = tol(:)'; % Required accuracy

% Init registration parameters
q = zeros(1,Nm*nq);

% Get image scaling from RMM/GMM fit
%-----------------------

scl = GetScaling(Nii,show_fit);

% Start coarse-to-fine
%-----------------------

for iter=1:numel(samp) % loop over sampling factors
    
    % Sampling level
    samp_i = samp(iter);
        
    % Init dat
    cl  = cell([1 Nm]);
    dat = struct('fix',struct('z',[],'y',[],'mat',[]), ...
                 'mov',struct('z',cl,'mat',cl), ...
                 'C',C, 'nq',nq);

    % Add fixed
    [z,mat]     = GetFeatures('SqrdGradMag',Nii(ixf),scl(ixf),samp_i);
    dat.fix.z   = z;
    dat.fix.mat = mat;
    dat.fix.y   = IdentityJittered(dat.fix);
    clear z
    
    % Add moving
    for c=1:Nm
        [z,mat]        = GetFeatures('SqrdGradMag',Nii(ixm(c)),scl(ixm(c)),samp_i);
        dat.mov(c).z   = z;
        dat.mov(c).mat = mat;
    end
    clear z
    
    % Initial search values and stopping criterias      
    sc = [];
    for c=1:Nm, sc = [sc sc0]; end
    iq = diag(sc*20/iter); % decrease stopping criteria w. iteration...     

    if show_align
        % Alignment before registration
        [cost,njtv] = CostFun(q,dat,show_align);
        ShowAlignment(njtv,cost,false);
    end
    
    % Start Powell
    q = spm_powell(q(:),iq,sc,mfilename,dat,show_align);

    if show_align
        % Alignment after registration
        [cost,njtv] = CostFun(q,dat,show_align);
        ShowAlignment(njtv,cost);
    end
end

% Get outputs
%-----------------------

% Transformations
R = repmat(eye(4),[1 1 C]);
for c=1:numel(dat.mov), R(:,:,ixm(c)) = GetRigid(q,c,dat); end

% Set output
res = struct('R',R,'ix_fixed',ixf,'ix_moving',ixm);
%==========================================================================

%==========================================================================
function [cost,njtv] = CostFun(q,dat,show_align)
C     = dat.C;
Mfix  = dat.fix.mat;
yfix  = dat.fix.y;

% Compute for fixed
njtv  = -sqrt(dat.fix.z)/C;
mtv   = dat.fix.z;

% Compute for moving
for c=1:numel(dat.mov) % loop over moving images
    
    % Get rigid transformation matrix from lie parameteristaion
    R = GetRigid(q,c,dat);
    
    % Make alignment vector field (y)
    Mmov = dat.mov(c).mat;
    M    = Mmov\R*Mfix;
    y    = Affine(yfix,M);
    
    % Move squared gradient magnitude (z)
    z               = spm_diffeo('bsplins',dat.mov(c).z, y, [ones(1,3)  0*ones(1,3)]);
    z(~isfinite(z)) = 0;
    
    % Add to voxel-wise cost
    mtv  = mtv + z;
    njtv = njtv - sqrt(z)/C;        
end    

% Get final cost
njtv = njtv + sqrt(mtv/C);
cost = sum(sum(sum(njtv,'double'),'double'),'double');

if show_align > 1, ShowAlignment(njtv,cost); end
%==========================================================================

%==========================================================================
function [z,mat] = GetFeatures(typ,Nii,scl,samp)

% Get image data, and possible down-sample
[im,mat] = GetImg(Nii,samp);

% Compute feature
z  = single(im);
vx = sqrt(sum(mat(1:3,1:3).^2));
if strcmp(typ,'SqrdGradMag')
    % Compute squared gradient magnitudes
    z = scl*Grad(z,vx);
    z = sum(z.^2,4);
    
    % Smooth a little bit
    filt = [0.125 0.75 0.125];    
    filt = filt./vx;
    z    = convn(z,reshape(filt,[3,1,1]),'same');
    z    = convn(z,reshape(filt,[1,3,1]),'same');
    z    = convn(z,reshape(filt,[1,1,3]),'same'); 
end
%==========================================================================

%==========================================================================
function ShowAlignment(vx_cost,sm_cost,show_align)
if nargin < 3, show_align = true; end
dm = [size(vx_cost) 1];
SetFigure('(SPM) Alignment',~show_align); 
if dm(3) == 1
    if show_align, cnt = 2;
    else,          cnt = 1;
    end
    subplot(2,1,cnt)
    imagesc(vx_cost'); axis off xy image;
    title(['cost_' num2str(show_align) ' = ' num2str(sm_cost)])
else        
    if show_align, cnt = [2 4 6];
    else,          cnt = [1 3 5];
    end
    subplot(3,2,cnt(1))
    sqgrd = vx_cost(:,:,round(dm(3)/2));        
    imagesc(sqgrd'); axis off xy image;
    title(['cost_' num2str(show_align) ' = ' num2str(sm_cost)])
    subplot(3,2,cnt(2))
    sqgrd = squeeze(vx_cost(:,round(dm(2)/2),:));        
    imagesc(sqgrd'); axis off xy image;    
    subplot(3,2,cnt(3))
    sqgrd = squeeze(vx_cost(round(dm(1)/2),:,:));        
    imagesc(sqgrd'); axis off xy image;
end
colormap(hot);
drawnow;
%==========================================================================

%==========================================================================
function y = Affine(y,mat)
dm = size(y);
y  = reshape(reshape(y,[prod(dm(1:3)) 3])*mat(1:3,1:3)' + mat(1:3,4)',[dm(1:3) 3]);
if dm(3) == 1, y(:,:,:,3) = 1; end
%==========================================================================

%==========================================================================
function g = Grad(im,vx,y)
d = [size(im) 1 1];
if nargin < 4, y = Identity(d(1:3)); end
g                = zeros([d(1:3) 3],'single');
% % 'sobel', 'prewitt', 'central', 'intermediate' 
% [g(:,:,:,1),g(:,:,:,2),g(:,:,:,3)] = imgradientxyz(im,'central');
% g(:,:,:,1)                         = g(:,:,:,1)/vx(1);
% g(:,:,:,2)                         = g(:,:,:,2)/vx(2);
% g(:,:,:,3)                         = g(:,:,:,3)/vx(3);
bc                 = 1;
[~,g(:,:,:,1),~,~] = spm_diffeo('bsplins',im,y, [2 0 0 bc*ones(1,3)]);
[~,~,g(:,:,:,2),~] = spm_diffeo('bsplins',im,y, [0 2 0 bc*ones(1,3)]);
[~,~,~,g(:,:,:,3)] = spm_diffeo('bsplins',im,y, [0 0 2 bc*ones(1,3)]);
g(:,:,:,1)         = g(:,:,:,1)/vx(1);
g(:,:,:,2)         = g(:,:,:,2)/vx(2);
g(:,:,:,3)         = g(:,:,:,3)/vx(3);
g(~isfinite(g))    = 0;
%==========================================================================

%==========================================================================
function id = Identity(d)
d  = [d(:)' 1 1];
d  = d(1:3);
id = zeros([d,3],'single');
[id(:,:,:,1),id(:,:,:,2),id(:,:,:,3)] = ndgrid(single(1:d(1)),single(1:d(2)),single(1:d(3)));
%==========================================================================

%==========================================================================
function R = GetRigid(q,c,dat)
nq = dat.nq;
B  = RigidBasis;

ix = (c - 1)*nq + 1:(c - 1)*nq + nq;        
qc = q(ix);
if nq == 3, qc = [qc(1) qc(2) 0 0 0 qc(3)];
end
R = spm_dexpm(qc,B); 
%==========================================================================

%==========================================================================
function y = IdentityJittered(fix)
dm = [size(fix.z) 1]; 
y  = Identity(dm);
rng('default') 
rng(1);  
y  = y + rand(size(y),'single');
%==========================================================================

%==========================================================================
function B = RigidBasis
B                = zeros(4,4,6);
B(1,4,1)         = 1;
B(2,4,2)         = 1;
B(3,4,3)         = 1;
B([2,3],[2,3],4) = [0 1;-1 0];    
B([3,1],[3,1],5) = [0 1;-1 0];
B([1,2],[1,2],6) = [0 1;-1 0];
%========================================================================== 

%==========================================================================
function SetFigure(figname,do_clear)
if nargin < 2, do_clear = false; end

f = findobj('Type', 'Figure', 'Name', figname);
if isempty(f)
    f = figure('Name', figname, 'NumberTitle', 'off');
end
set(0, 'CurrentFigure', f);   
if do_clear
    clf(f);
end
%==========================================================================

%==========================================================================
function [noise,mu_brain,mu_bg] = FitGMM(Nii,speak,K)
if nargin < 2, speak = false; end
if nargin < 3, K     = 2; end
   
% Get image date (vectorised)
f               = Nii.dat(:);
f(f == 0)       = [];
f(~isfinite(f)) = [];
f(f == min(f))  = [];
f(f == max(f))  = [];

% Histogram bin voxels
x = min(f):max(f);
c = hist(f(:),x);

% Transpose
x = x';
c = c';

% Intensity distribution hyper-parameters
MU0 = mean(f)*ones([1 K]) + K*randn([1 K]);
b0  = ones([1 K]);
n0  = ones([1 K]);
V0  = ones([1 1 K]);

% Fit GMM
[~,MU,A,PI] = spm_gmm(x,K,c,'BinWidth',1,'GaussPrior',{MU0,b0,V0,n0}, ...
                      'Tolerance',1e-6,'Start','prior','Verbose',0,'Prune',true);

% Compute variance for each class
V = zeros(size(A));
for k=1:size(V,3)
    V(:,:,k) = inv(A(:,:,k));
end

% Get standard deviation of class closest to air (-1000)
sd       = sqrt(squeeze(V));
noise    = min(sd);
[~,ix]   = max(MU);
mu_brain = MU(ix);
[~,ix]   = min(MU);
mu_bg    = MU(ix);

if speak
    % Plot histogram + GMM fit
    p = zeros([numel(x) K],'single');
    for k=1:K
        p(:,k) = PI(k)*mvnpdf(x(:),MU(:,k),V(:,:,k));
    end
    sp = sum(p,2) + eps;    
    md = mean(diff(x));
    
%     plot(x(:),p,'--',x(:),c/sum(c)/md,'b.',x(:),sp,'r','LineWidth',2); 
    plot(x(:),c/sum(c)/md,'b.',x(:),sp,'r',x(:),p,'--','LineWidth',2); drawnow
    drawnow
    
    FontSize = 10;
    set(gcf, 'Color', 'w')
    set(gca,'FontSize',FontSize)
    legend('Empirical','Mixture fit','Air class','Brain class')
%     xlabel('Image intensity')
    title('K=2 GMM Fit')
    axis tight
end
%==========================================================================

%==========================================================================
function [img,mat,dm] = GetImg(Nii,samp,deg,bc)
if nargin < 2, samp = 1; end
if nargin < 3, deg  = 0; end
if nargin < 4, bc   = 0; end

% Input image properties
img   = Nii.dat(:,:,:);    
mat0  = Nii.mat;
vx    = sqrt(sum(mat0(1:3,1:3).^2));
dm0   = size(img);
dm0   = [dm0 1];

if samp == 0    
    mat = mat0;
    dm  = dm0(1:3);
else
    samp = max([1 1 1],ceil(samp*[1 1 1]./vx));

    D    = diag([1./samp 1]);
    mat  = mat0/D;
    dm   = floor(D(1:3,1:3)*dm0(1:3)')';

    if dm0(3) == 1
        dm(3) = 1;
    end

    % Make interpolation grid
    [x0,y0,z0] = ndgrid(1:dm(1),1:dm(2),1:dm(3));

    T = mat0\mat;    

    x1 = T(1,1)*x0 + T(1,2)*y0 + T(1,3)*z0 + T(1,4);
    y1 = T(2,1)*x0 + T(2,2)*y0 + T(2,3)*z0 + T(2,4);
    z1 = T(3,1)*x0 + T(3,2)*y0 + T(3,3)*z0 + T(3,4);

    if dm0(3) == 1
        z1 = ones((size(z1)));
    end

    % Resample
    if numel(deg)  == 1, deg  = deg*ones([1 3]);  end
    if numel(bc)   == 1, bc   = bc*ones([1 3]);   end

    par                 = [deg bc];
    img                 = spm_bsplins(spm_bsplinc(img,par),x1,y1,z1,par);    
    img(~isfinite(img)) = 0;
end
%==========================================================================

%==========================================================================
function scl = GetScaling(Nii,show_fit)
C   = numel(Nii);
scl = zeros(1,C); % scaling
nr  = floor(sqrt(C));
nc  = ceil(C/nr);  
for c=1:C
    if show_fit
        if c == 1, SetFigure('(SPM) Scaling',true); 
        else,      SetFigure('(SPM) Scaling'); 
        end 
        subplot(nr,nc,c); 
    end        

    isneg = min(Nii(c).dat(:)) < 0;    
    if isneg
        % Fit GMM
        [~,mu_brain,mu_bg] = FitGMM(Nii(c),show_fit);         
        scl(c)             = 1/(mu_brain - mu_bg);
    else
        % Fit RMM        
        [~,mu_brain] = spm_noise_estimate_mod(Nii(c),show_fit);
        scl(c)       = 1/mu_brain;
    end       
end
%==========================================================================

%==========================================================================
function [noise,mu_brain] = spm_noise_estimate_mod(Scans,speak,nr,nc,cnt_subplot)
% Estimate avarage noise from a series of images
% FORMAT noise = spm_noise_estimate(Scans)
% Scans - nifti structures or filenames of images
% noise - standard deviation estimate
% _______________________________________________________________________
%  Copyright (C) 2018 Wellcome Trust Centre for Neuroimaging

if nargin < 2, speak       = 0; end
if nargin < 3, nr          = 0; end
if nargin < 4, nc          = 0; end
if nargin < 5, cnt_subplot = 0; end

if ~isa(Scans,'nifti'), Scans = nifti(Scans); end

noise    = zeros(numel(Scans),1);
mu_brain = zeros(numel(Scans),1);
for i=1:numel(Scans)
    Nii = Scans(i);
    f   = Nii.dat(:,:,:);
    if spm_type(Nii.dat.dtype(1:(end-3)),'intt')
        f(f==max(f(:))) = 0;
        x      = 0:Nii.dat.scl_slope:max(f(:));
        [h,x]  = hist(f(f>0),x);
    else
        x      = (0:1023)*(max(f(:))/1023);
        f(f==max(f(:))) = 0;
        [h,x]  = hist(f(f>0 & isfinite(f)),x);
    end
    [mg,nu,sd,mu] = spm_rice_mixture_mod(double(h(:)),double(x(:)),2,speak,nr,nc,cnt_subplot + i);
    
    noise(i)    = min(sd);
    
    x           = -nu.^2./(2*sd.^2);
    msk         = x>-20;
    Laguerre    = exp(x(msk)/2).*((1-x(msk)).*besseli(0,-x(msk)/2)-x(msk).*besseli(1,-x(msk)/2));
    Ey( msk)    = sqrt(pi*sd(msk).^2/2).*Laguerre;
    Ey(~msk)    = nu(~msk);
    mu_brain(i) = max(Ey);
end
%==========================================================================

%==========================================================================
function [mg,nu,sig,mu] = spm_rice_mixture_mod(h,x,K,speak,nr,nc,cnt_subplot)
% Fit a mixture of Ricians to a histogram
% FORMAT [mg,nu,sig] = rice_mixture(h,x,K)
% h    - histogram counts
% x    - bin positions (plot(x,h) to see the histogram)
% K    - number of Ricians
% mg   - integral under each Rician
% nu   - "mean" parameter of each Rician
% sig  - "standard deviation" parameter of each Rician
% mu   - "mean" parameter of each Rician, from sufficient statistics
%
% An EM algorithm is used, which involves alternating between computing
% belonging probabilities, and then the parameters of the Ricians.
% The Koay inversion technique is used to compute the Rician parameters
% from the sample means and standard deviations. This is described at
% http://en.wikipedia.org/wiki/Rician_distribution
%_______________________________________________________________________
% Copyright (C) 2018 Wellcome Trust Centre for Neuroimaging

mg  = ones(K,1)/K;
nu  = (0:(K-1))'*max(x)/(K+1);
sig = ones(K,1)*max(x)/(10*K);
mu  = zeros(K,1);

m0 = zeros(K,1);
m1 = zeros(K,1);
m2 = zeros(K,1);
ll = -Inf;
for iter=1:10000
    p  = zeros(numel(x),K);
    for k=1:K
        % Product Rule
        % p(class=k, x | mg, nu, sig) = p(class=k|mg) p(x | nu, sig, class=k)
        p(:,k) = mg(k)*ricepdf(x(:),nu(k),sig(k)^2);
    end

    % Sum Rule
    % p(x | mg, nu, sig) = \sum_k p(class=k, x | mg, nu, sig)
    sp  = sum(p,2)+eps;
    oll = ll;
    ll  = sum(log(sp).*h(:)); % Log-likelihood
    if ll-oll<1e-8*sum(h), break; end

%     fprintf('%g\n',ll);
%     md = mean(diff(x));
%     plot(x(:),p,'--',x(:),h/sum(h)/md,'b.',x(:),sp,'r'); drawnow

    % Bayes Rule
    % p(class=k | x, mg, nu, sig) = p(class=k, x | mg, nu, sig) / p(x | mg, nu, sig)
    p = bsxfun(@rdivide,p,sp);

    % Compute moments from the histograms, weighted by the responsibilities (p).
    for k=1:K
        m0(k) = sum(p(:,k).*h(:));              % Number of voxels in class k
        m1(k) = sum(p(:,k).*h(:).*x(:));        % Sum of the intensities in class k
        m2(k) = sum(p(:,k).*h(:).*x(:).*x(:));  % Sum of squares of intensities in class k
    end

    mg = m0/sum(m0); % Mixing proportions
    for k=1:K
        mu1 = m1(k)./m0(k);                                % Mean 
        mu2 = (m2(k)-m1(k)*m1(k)/m0(k)+1e-6)/(m0(k)+1e-6); % Variance

        % Compute nu & sig from mean and variance
        [nu(k),sig(k)] = moments2param(mu1,mu2);
        
        mu(k) = mu1;
    end
    %disp([nu'; sig'])
end

if speak
    if nr > 0
        subplot(nr,nc,cnt_subplot);
    end
    
    md = mean(diff(x));
    plot(x(:),h/sum(h)/md,'b.',x(:),sp,'r',x(:),p,'--','LineWidth',2); drawnow
    
    FontSize = 10;
    set(gcf, 'Color', 'w')
    set(gca,'FontSize',FontSize)
    legend('Empirical','Mixture fit','Air class','Brain class')
%     xlabel('Image intensity')
    title('K=2 RMM Fit')
end
%==========================================================================

%==========================================================================
function [nu,sig] = moments2param(mu1,mu2)
% Rician parameter estimation (nu & sig) from mean (mu1) and variance
% (mu2) via the Koay inversion technique.
% This follows the scheme at
% https://en.wikipedia.org/wiki/Rice_distribution#Parameter_estimation_.28the_Koay_inversion_technique.29
% This Wikipedia description is based on:
% Koay, C.G. and Basser, P. J., Analytically exact correction scheme
% for signal extraction from noisy magnitude MR signals,
% Journal of Magnetic Resonance, Volume 179, Issue = 2, p. 317–322, (2006)

r     = mu1/sqrt(mu2);
theta = sqrt(pi/(4-pi));
if r>theta
    for i=1:256
        xi    = 2+theta^2-pi/8*exp(-theta^2/2)*((2+theta^2)*besseli(0,theta^2/4)+theta^2*besseli(1,theta^2/4))^2;
        g     = sqrt(xi*(1+r^2)-2);
        if abs(theta-g)<1e-6, break; end
        theta = g;
    end
    sig = sqrt(mu2)/sqrt(xi);
    nu  = sqrt(mu1^2+(xi-2)*sig^2);
else
    nu  = 0;
    sig = (2^(1/2)*(mu1^2 + mu2)^(1/2))/2;
end
%==========================================================================

%==========================================================================
function p = ricepdf(x,nu,sig2)
% Rician PDF
% p = ricepdf(x,nu,sig2)
% https://en.wikipedia.org/wiki/Rice_distribution#Characterization
p       = zeros(size(x));
tmp     = -(x.^2+nu.^2)./(2*sig2);
msk     = (tmp > -95) & (x*(nu/sig2) < 85) ; % Identify where Rice probability can be computed
p(msk)  = (x(msk)./sig2).*exp(tmp(msk)).*besseli(0,x(msk)*(nu/sig2)); % Use Rician distribution
p(~msk) = (1./sqrt(2*pi*sig2))*exp((-0.5/sig2)*(x(~msk)-nu).^2);      % Use Gaussian distribution
%==========================================================================