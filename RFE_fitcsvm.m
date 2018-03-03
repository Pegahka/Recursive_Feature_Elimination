
% original code by PKF
% RFE original courtesy of KE YAN, SM
% https://ch.mathworks.com/matlabcentral/fileexchange/50701-feature-selection-with-svm-rfe



clear all
close all
clc;

classes = [1,2,3]; % possible classes/labels
load CP-allGroups;

features_1 = features(labels==classes(1),:);
features_2 = features(labels==classes(3),:);
    
     
%% random subSampling
p = min(size(features_1,1), size(features_2,1));
idx = randsample(1:size(features_1,1),p);
features_1 = features_1(idx,:);
idy = randsample(1:size(features_2,1),p);
features_2 = features_2(idy,:);
features = [features_1;features_2]; 

%% binarize labels
labels = [];
labels(1:p,:) = 1;
labels(p+1:2*p,:) = 0;
labels = logical(labels);
    
    
%% parameters RFE & Classification    
numFeat = 38; % select the first numFeat highest ranked features
nrFolds = 10; %number of folds of crossvalidation, 10 is standard
kernel = 'linear'; % 'linear', 'rbf' or 'polynomial'
C = 1; % C is the 'boxconstraint' parameter. Small C = Allow for more missclassif.
solver = 'L1QP';
nrRand = 10; % at least equal to 2 for error-calculation


for k = 1:nrRand
      
    cvFolds = crossvalind('Kfold', labels, nrFolds);
        
for i = 1:nrFolds                            % iteratre through each fold
    testIdx = (cvFolds == i);                % indices of test instances
    trainIdx = ~testIdx;                     % indices training instances
    
    param.kertype = 0;
    ranking(i,:) = ftSel_SVMRFECBR(features(trainIdx,:),labels(trainIdx), param);

% compare different nr of Features for same train set

for nF = 1:numFeat
    %train the SVM
    cl = fitcsvm(features(trainIdx,ranking(i,1:nF)), labels(trainIdx),'KernelFunction',kernel,'Standardize',true,...
    'BoxConstraint',C,'ClassNames',[0,1],'Solver',solver);
             
    [label,scores] =  predict(cl, features(testIdx,ranking(i,1:nF)));
    eq = sum(label==labels(testIdx));
    accuracy(i) = eq/numel(labels(testIdx));
    crossValAcc(i,nF) = mean(accuracy);
end

end

accRFE(k,:) = mean(crossValAcc); % average crossvalidation accuracy for each iteration
rankingAll(i,:) = mode(ranking);
end

%% plotting number of Features vs Accuracy & std(Accuracy)

RFEaccuracy = mean(accRFE);
RFEstdAcc = std(accRFE);
x = 1:numFeat;
errorbar(x,RFEaccuracy,RFEstdAcc)
xlabel('Number of highest-ranked Features')
ylab = sprintf('Classification accuracy and error over %d iterations',nrRand);
ylabel(ylab)
ylim([0.5 1])
xlim([1 numFeat])