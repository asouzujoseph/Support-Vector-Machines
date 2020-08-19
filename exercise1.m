%% SVM Exercise Session 1
%% Two Guassian example
%% generate random data sets and insert a line to separate the two classes
% X1 = randn (50 ,2) + 1;
% X2 = randn (51 ,2) - 1;
% Y1 = ones (50 ,1);
% Y2 = -ones (51 ,1);
% figure ;
% hold on;
% plot (X1 (: ,1) , X1 (: ,2) , 'ro'); 
% plot (X2 (: ,1) , X2 (: ,2) , 'bo');
% hold off;
% line([-1.5, 1.2], [3, -2])

%% Train the LS-SVM classifier using polynomial kernel
%% The objective is to check the impact of different polynomial degrees on error rate 
load iris
type = 'c';
gam = 1;
t = 1;
deglist = [1,2,3,4,5,6,7,8,9,10];
errlist=[];
for degree=deglist
    disp(['degree :', num2str(degree)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end
figure;
plot(deglist, errlist, '*-'), 
xlabel('degree'), ylabel('number of misclassification errors'),


%% Train the LS-SVM classifier using RBF kernel
%% The objective is to select a good value for sigma squared while fixing gam to 1
load iris    
disp('RBF kernel')
type = 'classification' 
gam = 1; sig2list=[0.01, 0.1,0.5,1,3,5,10];
errlist=[];
for sig2=sig2list
    disp(['sig2 : ', num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause, 
end

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(sig2list), errlist, '*-'), 
xlabel('log(sig2)'), ylabel('number of misclassification errors'),


%% Train the LS-SVM classifier using RBF kernel
%% The objective is to select a good value for gam while fixing sigma squared to 1
sig2 = 1; gamList=[0.02,0.1,0.5,1,3,5,10,40];
errlist=[];
for gam=gamList   
    disp(['gam : ', num2str(gam)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,         
end

%
% make a plot of the misclassification rate wrt. sig2
%
figure;
plot(log(gamList), errlist, '*-'), 
xlabel('log(gam)'), ylabel('number of misclassification errors'),



%% Compute performance on a range of gam and sig2 values
load iris
type='classification';
errlist=[];
LOOCList=[];
foldList=[];
randomList=[];
gamList=[0.001,0.01,0.1,0.5,1,5,10,100,500,1000];
sig2list=[0.001,0.01,0.1,0.5,1,5,10,100,500,1000];
for gam=gamList
    sig2=gam;
    disp(['gam : ', num2str(gam),' sig2 : ',num2str(sig2)]),
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

    % Plot the decision boundary of a 2-d LS-SVM classifier
    plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});

    % Obtain the output of the trained classifier
    [Yht, Zt] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);
    err = sum(Yht~=Ytest); errlist=[errlist; err];
    fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Ytest)*100)
    disp('Press any key to continue...'), pause,
    LOOC=leaveoneout({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},'misclass');
    random=rsplitvalidate({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},0.80,'misclass');
    cross=crossvalidate({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},10,'misclass');
    foldList=[foldList;cross];
    randomList=[randomList;random];
    LOOCList=[LOOCList;LOOC];
end
figure;
hold on;
plot(log(gamList), LOOCList, 'g'),
plot(log(gamList), randomList, 'r--o'),
plot(log(gamList), foldList, 'b--o'),
hold off;
title('Performance measures');
legend('Leave one out validation','random split method','10-fold crossvalidation');
xlabel('log(gam) or log(sig2)'), ylabel('misclassification rate'),


%% Automatic parameter tuning
%% try different types of automatic parameter tuning algorithms
load iris
[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'misclass'});
disp('Press any key to continue...'), pause,
[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});


%% ROC curves
%% generate ROC curves on the iris dataset and use tuned gam / sig2 values
load iris
type = 'classification';
% Tune gam and sig2
[gam ,sig2] = tunelssvm ({ Xtrain , Ytrain , 'c', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);

%% Bayesian inference for classification
% the values of gam and sigma2 from the ROC section are still used here
% It is recommended to initiate the model with appropriate starting values:
[gam, sig2] = bay_initlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
% Optimization on the second level leads to an optimal regularization parameter
[model, gam_opt] = bay_optimize({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},2);
%Optimization on the third level leads to an optimal kernel parameter:
[cost_L3,sig2_opt] = bay_optimize({Xtrain,Ytrain,'c',gam_opt,sig2,'RBF_kernel'},3);
% use the values of the optimized sig2 and gam values to get posterior
%Ymodout = bay_modoutClass({Xtrain,Ytrain,'c',gam_opt,sig2_opt,'RBF_kernel'},'figure');

Ymodout = bay_modoutClass({Xtrain,Ytrain,'c',0.496,7.7543,'RBF_kernel'},'figure');


%%Homework problems
%% Ripley dataset
%% visualize training data
load ripley
gscatter(Xtrain(:,1),Xtrain(:,2),Ytrain(:,:),'kr','o');
title("Visualization of Ripley Training Dataset");
disp("press any key to visualize the test set"),pause,
gscatter(Xtest(:,1),Xtest(:,2),Ytest(:,:),'kr','o');
title("Visualization of Ripley Test Dataset");

%% Using linear, polynomial and RBF models with tuned hyperparameters and kernel parameters
%% compute the ROC curves and select the best model
%% Linear kernel
clc
clear
load ripley
type = 'classification';
L_fold = 10; % L-fold crossvalidation
% tune gam
gam = tunelssvm({Xtrain,Ytrain,type,[],[],'lin_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});
figure; plotlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);


%% Polynomial kernel
clc
clear
load ripley
type = 'classification';
t = 1;
degree = 4; 
L_fold = 10; % L-fold crossvalidation
% Tune gam and sig2
[gam,~] = tunelssvm({Xtrain,Ytrain,type,[],[],'poly_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'});
figure; plotlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,[t; degree],'poly_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);


%% RBF model
clc
clear
load ripley
type = 'classification';
% tune gam and sig2
[gam,sig2] = tunelssvm({Xtrain,Ytrain,type,[],[],'RBF_kernel'}, 'simplex','crossvalidatelssvm',{10,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
figure; plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b},Xtest);
roc(Y_latent,Ytest);



%% Wisconsin Breast Cancer Dataset
%& visualize data
clc
clear
load breast
gscatter(trainset(:,1),trainset(:,2),labels_train(:,:),'kr','o');
title("Visualization of Breast Cancer Training Dataset");
disp("press any key to visualize the test set"),pause,
gscatter(testset(:,1),testset(:,2),labels_test(:,:),'kr','o');
title("Visualization of Breast Cancer Test Dataset");

%% Using linear, polynomial and RBF models with tuned hyperparameters and kernel parameters
%% compute the ROC curves and select the best model
%% Linear kernel
clc;
clear;
load breast
type = 'classification';
L_fold = 10; % L-fold crossvalidation
% tune gam
gam = tunelssvm({trainset,labels_train,type,[],[],'lin_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({trainset,labels_train,type,gam,[],'lin_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
Y_latent = latentlssvm({trainset,labels_train,type,gam,[],'lin_kernel'},{alpha,b},testset);
roc(Y_latent,labels_test);

%% Polynomial kernel
clc;
clear;
load breast
type = 'classification';
t = 1;
degree = 4; 
L_fold = 10; % L-fold crossvalidation
% Tune gam and sig2
[gam,~] = tunelssvm({trainset,labels_train,type,[],[],'poly_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({trainset,labels_train,type,gam,[t; degree],'poly_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({trainset,labels_train,type,gam,[t; degree],'poly_kernel'},{alpha,b},testset);
roc(Y_latent,labels_test);


%% RBF model
clc;
clear;
load breast
type = 'classification';
% tune gam and sig2
[gam,sig2] = tunelssvm({trainset,labels_train,type,[],[],'RBF_kernel'}, 'simplex','crossvalidatelssvm',{10,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'},{alpha,b},testset);
roc(Y_latent,labels_test);


%% Diabetes dataset
% visualize data
clc;
clear;
load diabetes
gscatter(trainset(:,1),trainset(:,2),labels_train(:,:),'kr','o');
title("Visualization of Diabetes Training Dataset");
disp("press any key to visualize the test set"),pause,
gscatter(testset(:,1),testset(:,2),labels_test(:,:),'kr','o');
title("Visualization of Diabetes Test Dataset");

%% Linear kernel
clc;
clear;
load diabetes
type = 'classification';
L_fold = 10; % L-fold crossvalidation
% tune gam
gam = tunelssvm({trainset,labels_train,type,[],[],'lin_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({trainset,labels_train,type,gam,[],'lin_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,[],'lin_kernel','preprocess'},{alpha,b});
Y_latent = latentlssvm({trainset,labels_train,type,gam,[],'lin_kernel'},{alpha,b},testset);
roc(Y_latent,labels_test);


%% Polynomial kernel
clc;
clear;
load diabetes
type = 'classification';
t = 1;
degree = 4; 
L_fold = 10; % L-fold crossvalidation
% Tune gam and sig2
[gam,~] = tunelssvm({trainset,labels_train,type,[],[],'poly_kernel'},'simplex','crossvalidatelssvm',{L_fold,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({trainset,labels_train,type,gam,[t; degree],'poly_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({trainset,labels_train,type,gam,[t; degree],'poly_kernel'},{alpha,b},testset);
roc(Y_latent,labels_test);

%% RBF model
clc;
clear;
load diabetes
type = 'classification';
% tune gam and sig2
[gam,sig2] = tunelssvm({trainset,labels_train,type,[],[],'RBF_kernel'}, 'simplex','crossvalidatelssvm',{10,'misclass'});
% Train the classification model.
[alpha,b] = trainlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'});
figure; plotlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
% latent variables are needed to make the ROC curve
Y_latent = latentlssvm({trainset,labels_train,type,gam,sig2,'RBF_kernel'},{alpha,b},testset);
roc(Y_latent,labels_test);