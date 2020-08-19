%% Kernel PCA
% change the number of PC
clc;
clear;
nb=400;
sig=0.3;
nb=nb/2;
% construct data
leng = 1;
for t=1:nb, 
  yin(t,:) = [2.*sin(t/nb*pi*leng) 2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  yang(t,:) = [-2.*sin(t/nb*pi*leng) .45-2.*cos(.61*t/nb*pi*leng) (t/nb*sig)]; 
  samplesyin(t,:)  = [yin(t,1)+yin(t,3).*randn   yin(t,2)+yin(t,3).*randn];
  samplesyang(t,:) = [yang(t,1)+yang(t,3).*randn   yang(t,2)+yang(t,3).*randn];
end

% plot dataset
h=figure; hold on
plot(samplesyin(:,1),samplesyin(:,2),'o');
plot(samplesyang(:,1),samplesyang(:,2),'o');
xlabel('X_1');
ylabel('X_2');
title('Structured dataset');
disp('Press any key to continue');
pause;
% get user inputs
ncList=[6,9,12,15,18,21];
for nc=ncList
    disp(nc);
    sig2=0.4;
    approx='eigs';
    % calculate the eigenvectors in the feature space (principal components)
    [lam,U] = kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
    % denoise
    xd = denoise_kpca([samplesyin;samplesyang],'RBF_kernel',sig2,[],approx,nc);
    plot(samplesyin(:,1),samplesyin(:,2),'o');
    plot(samplesyang(:,1),samplesyang(:,2),'o');
    plot(xd(:,1),xd(:,2),'r+');
    title('Kernel PCA - Denoised datapoints in red');
    disp("press key to see next"),pause,
end

%% linear PCA
% Projections on the first component using linear PCA

dat=[samplesyin;samplesyang];
dat(:,1)=dat(:,1)-mean(dat(:,1));
dat(:,2)=dat(:,2)-mean(dat(:,2));


[lam_lin,U_lin] = pca(dat);


%proj_lin=grid*U_lin;

figure;

plot(samplesyin(:,1),samplesyin(:,2),'o');hold on;
plot(samplesyang(:,1),samplesyang(:,2),'o');
%contour(Xax,Yax,reshape(proj_lin(:,1),length(Yax),length(Xax)));

xdl=U_lin(:,1)*U_lin(:,1)';
plot(xdl(:,1),xdl(:,2),'r+');

title('Linear PCA - Denoised data points using the first principal component');



%% Spectral PCA
clear;
load two3drings;        % load the toy example
[N,d]=size(X);
perm=randperm(N);   % shuffle the data
X=X(perm,:);
sig2List=[0.001,0.005,0.01,0.02,0.05];  % set the kernel parameters

for sig2=sig2List
    K=kernel_matrix(X,'RBF_kernel',sig2);   %compute the RBF kernel (affinity) matrix
    D=diag(sum(K));         % compute the degree matrix (sum of the columns of K)
    [U,lambda]=eigs(inv(D)*K,3);  % Compute the 3 largest eigenvalues/vectors using Lanczos
                                  % The largest eigenvector does not contain
                                  % clustering information. For binary clustering,
                                  % the solution is the second largest eigenvector.
    clust=sign(U(:,2)); % Threshold the eigenvector solution to obtain binary cluster indicators
    [y,order]=sort(clust,'descend');    % Sort the data using the cluster information
    Xsorted=X(order,:);
    Ksorted=kernel_matrix(Xsorted,'RBF_kernel',sig2);   % Compute the kernel matrix of the
                                                        % sorted data.
    proj=K*U(:,2:3);    % Compute the projections onto the subspace spanned by the second,
                        % and third largest eigenvectors.
    %%%% PLOTTING SECTION %%%%                                                 
    subplot(1,2,1)
    scatter3(X(:,1),X(:,2),X(:,3),15);
    title('Two interlaced rings in a 3D space');
    subplot(1,2,2);
    scatter3(X(:,1),X(:,2),X(:,3),30,clust);
    title('Clustering results');
    disp('<<<<<<<<<<<<Press any key>>>>>>>>>>>>>>');
    pause;
    figure;
    subplot(1,2,1);
    imshow(K);
    title('Kernel matrix of the original data');
    subplot(1,2,2);
    imshow(Ksorted);
    title('Kernel matrix after sorting the data using the cluster information');
    figure;
    scatter(proj(:,1),proj(:,2),15,clust);
    title('Projections onto subspace spanned by the 2nd and 3rd largest eigenvectors');
    disp('<<<<<<<<<<<<Press any key>>>>>>>>>>>>>>');
    pause;
end

%% Fixed size LS-SVM






















