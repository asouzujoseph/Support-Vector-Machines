%% LS_SVM regression model with RBF kernel
clc;
clear;
% load dataset or create dataset
X = linspace(-1,1,50)';
Y = (15*(X.^2-1).^2.*X.^4).*exp(-X)+normrnd(0,0.1,length(X),1);
type = 'function estimation';
% tuning parameters
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'simplex','leaveoneoutlssvm',{'mse'});
% train the model
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
% visualize model
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});
% evaluate model on test data. First create test data
Xt = rand(10,1).*sign(randn(10,1));
Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);
pause,
plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});


%% Reegression of the sinc function
% construct a LS-SVM regression model with the RBF kernel
% try out a range of different gam and sig2 parameters
clc;
clear;
type = 'function estimation';
sigList=[0.01,1,100];
gamList=[10,1000,1000000];
X = ( -3:0.01:3)';
Y = sinc (X) + 0.1.* randn ( length (X), 1);
Xtrain = X (1:2: end);
Ytrain = Y (1:2: end);
Xtest = X (2:2: end);
Ytest = Y (2:2: end);

for num = 1:length(sigList)
    gam=gamList(num);
    sig2=sigList(num);
    disp([gam,sig2])
    % train the model 
    [alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});
    % visualize model on training data
    %plotlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'},{alpha,b});
    % evaluate model on test data. 
    Ytest = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
    % visualize model on test data
    plotlssvm({Xtest,Ytest,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
    cross=crossvalidate({Xtest,Ytest,'f',gam,sig2,'RBF_kernel'},10,'mse');
    disp(["MSE is : ", cross]);
    disp("press any key to continue"), pause, 
end

%% Automatic parameter tuning
%% try different types of automatic parameter tuning algorithms
[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
disp('Press any key to continue...'), pause,
%[gam ,sig2 , cost ] = tunelssvm ({ Xtrain , Ytrain , 'f', [], [],'RBF_kernel'}, 'gridsearch', 'crossvalidatelssvm',{10, 'mse'});

%% Application of the Bayesian framework
sig2 = 0.4;
gam = 10;
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);
% optimize parameters at the three different level of bayesian framework
[~, alpha ,b] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 1);
[~, gamopt] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 2);
[~, sig2opt ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 }, 3);
% plot with optimized parameters
sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gamopt , sig2opt }, 'figure');
disp("press any key to move to next plot"), pause,
% plot with the usual parameters
sig2e2 = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 }, 'figure');

%% Automatic relevance determination
X = 6.* rand (100 , 3) - 3;
Y = sinc (X(: ,1)) + 0.1.* randn (100 ,1) ;
%use the Bayes optimized gam and sig2 values 
inputs = bay_lssvmARD({X,Y,'f', gamopt,sig2opt});
[alpha,b] = trainlssvm({X(:,inputs),Y,'f', 10,1});
plotlssvm({X,Y,'f',gamopt,sig2opt,'RBF_kernel'},{alpha,b});


%% Robust regression
% create dataset
X = ( -6:0.2:6)';
Y = sinc (X) + 0.1.* rand ( size (X));
% add outliers
out = [15 17 19];
Y(out) = 0.7+0.3* rand ( size ( out));
out = [41 44 46];
Y( out) = 1.5+0.2* rand ( size ( out));
% initialize model
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
L_fold = 10;
%tune model with appropriate weight
model = tunelssvm(model,'simplex','rcrossvalidatelssvm',{L_fold,'mae'},'wmyriad');
% robust regression
model = robustlssvm(model);
% plot to visualize
plotlssvm(model);

%% non-robust regression
%create dataset
X = ( -6:0.2:6)';
Y = sinc (X) + 0.1.* rand ( size (X));
% add outliers
out = [15 17 19];
Y(out) = 0.7+0.3* rand ( size ( out));
out = [41 44 46];
Y( out) = 1.5+0.2* rand ( size ( out));
% tune parameters
[gam ,sig2 ] = tunelssvm ({ X , Y , 'f', [], [],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm',{10, 'mse'});
% train model
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
% plot
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel'},{alpha,b});


%% Homework problems
%% Time series prediction - logmap dataset (non-optimized)
clc
clear
load logmap;
rng('default');
% map Z training dataset into regression problem using the command
% "windowize"
order =10;
X = windowize (Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );
% % build a model
gam = 10;
sig2 = 10;
[alpha,b] = trainlssvm ({X, Y, 'f', gam , sig2 });
% define starting point of the prediction
Xs = Z(end - order +1: end , 1); % usually the last point of the training dataset
nb = 50;  % equal to the number of datapoints in the test set.
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
% visualize the performance of the predictor
figure ;
hold on;
plot (Ztest , 'k'); % k == black color (actual data being predicted)
plot ( prediction , 'r'); % r == red (the prediction by model)
title(" order = 10, gam = 10, sig2 = 10");
legend("actual data","prediction",'Location','southwest');
xlabel("time index");
ylabel("amplitude");
hold off;

%% optimization / tuning of parameters on logmap dataset
clc;
clear;
load logmap;
rng('default');
% map dataset into regression problem
orderList=[1,10,20,30,40,50,60,70,80,90,100];
foldList=[];
errList=[];
for order=orderList
    disp(['order : ',num2str(order)])
    X = windowize (Z, 1:( order + 1));
    Y = X(:, end);
    X = X(:, 1: order );
    Xs = Z(end - order +1: end , 1);
    % % tune parameters
    [gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
    % train model
    [alpha,b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel'});
    % predict 
    nb = 50; % always equal to number of datapoints in test dataset
    prediction = predict({X,Y,'f',gam,sig2,'RBF_kernel'},Xs,nb);
    plot([prediction Ztest]);
    title(order);
    disp("next one"),pause,
    er = Ztest-prediction; 
    err=mae(er);
    errList=[errList; err];
end
plot(log(orderList), errList,'k--o');
title("prediction accuracy using automatically tuned gam and sig2");
ylabel("mean absolute error");
xlabel("log(order)");


%% Analysis on logmap dataset using the optimized settings
clc
clear
load logmap;
rng('default');
% map Z training dataset into regression problem using the command
% "windowize"
order =50;
X = windowize (Z, 1:( order + 1));
Y = X(:, end);
X = X(:, 1: order );
% % build a model
gam = 64.8;
sig2 = 6345.7;
[alpha,b] = trainlssvm ({X, Y, 'f', gam , sig2 });
% define starting point of the prediction
Xs = Z(end - order +1: end , 1); % usually the last point of the training dataset
nb = 50;  % equal to the number of datapoints in the test set.
prediction = predict ({X, Y, 'f', gam , sig2 }, Xs , nb);
% visualize the performance of the predictor
figure ;
hold on;
plot (Ztest , 'k'); % k == black color (actual data being predicted)
plot ( prediction , 'r'); % r == red (the prediction by model)
title(" order = 50, gam = 64.8, sig2 = 6345.7");
legend("actual data","prediction",'Location','southwest');
xlabel("time index");
ylabel("amplitude");
hold off;

 %% Santafe dataset
 %% example code in textbook
clc;
clear;
load santafe;
rng('default');
% map dataset into regression problem
order = 50;
Xu = windowize(Z,1:order+1);
Xtra = Xu(1:end-order,1:order); %training set
Ytra = Xu(1:end-order,end); %training set
Xs=Z(end-order+1:end,1); 
% tune parameters
[gam,sig2] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
% train model
[alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'});
% predict next 200 points (equivalent to size of test dataset
nb = 200; % always equal to number of datapoints in test dataset
prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs,nb);
figure ;
hold on;
plot (Ztest , 'k'); % k == black color (actual data being predicted)
plot ( prediction , 'b'); % r == red (the prediction by model)
title(" order = 50, gam = 1.042e+03, sig2 = 42");
legend("actual data","prediction",'Location','northeast');
xlabel("time index");
ylabel("amplitude");
hold off;

 
%% optimization run - santa fe dataset
clc;
clear;
load santafe;
rng('default');
% map dataset into regression problem
orderList=[10,20,30,40,50,60,70,80,90,100];
errList=[];

for order=orderList
    disp(['order : ',num2str(order)])
    Xu = windowize(Z,1:order+1);
    Xtra = Xu(1:end-order,1:order);
    Ytra = Xu(1:end-order,end);
    Xs=Z(end-order+1:end,1);
    % % tune parameters
    [gam,sig2] = tunelssvm({Xtra,Ytra,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm',{10,'mae'});
    % train model
    [alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'});
    % predict 
    nb = 200; % always equal to number of datapoints in test dataset
    prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs,nb);
    plot([prediction Ztest]);
    title(order);
    disp("next one"),pause,
    er = Ztest-prediction; 
    err=mae(er);
    errList=[errList; err];
    
end
plot(log(orderList), errList,'r--o');
title("prediction accuracy using automatically tuned gam and sig2");
ylabel("mean absolute error");
xlabel("log(order)");


%% Rerun of Santa fe dataset with the optimized parameters
clc;
clear;
load santafe;
rng('default');
% map dataset into regression problem
order = 70;
Xu = windowize(Z,1:order+1);
Xtra = Xu(1:end-order,1:order); %training set
Ytra = Xu(1:end-order,end); %training set
Xs=Z(end-order+1:end,1); 
gam = 297.23;
sig2 = 69.65;
% train model
[alpha,b] = trainlssvm({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'});
% predict next 200 points (equivalent to size of test dataset
nb = 200; % always equal to number of datapoints in test dataset
prediction = predict({Xtra,Ytra,'f',gam,sig2,'RBF_kernel'},Xs,nb);
figure ;
hold on;
plot (Ztest , 'k'); % k == black color (actual data being predicted)
plot ( prediction , 'b'); % r == red (the prediction by model)
title("Optimized parameters [order = 70, gam = 297.23, sig2 = 69.65]");
legend("actual data","prediction",'Location','northeast');
xlabel("time index");
ylabel("amplitude");
hold off;










