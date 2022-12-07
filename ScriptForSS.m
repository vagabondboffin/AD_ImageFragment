clear all
 
warning off

currentFolder = which('ScriptForNB.m');
currentFolder(strfind(currentFolder,'ScriptForNB.m')-1:end) = [];
AllPaths_FFC = genpath(currentFolder);
addpath(AllPaths_FFC);


load('normalDatasetPath');
DatasetN = Dataset;
NormalClassLabels = ClassLabels;

load('anomalyDatasetPath');
DatasetA_ = Dataset;


K = 5;
TV = [0 1 1];
TVTIndex = [0 1];
idx = TVTIndex(1)+(TVTIndex(2)-TVTIndex(1))*(0:1/K:1);

[ErrorMsg,KFoldsIdx] = Partition_Dataset_FFC(DatasetN(:,end-1:end),NormalClassLabels,{idx});
AllIndex = [];
for j=1:K
    AllIndex = union(AllIndex,KFoldsIdx{j});
end
PartitionGenerateError = [true true];

ADCnfMtrx = cell(1,K);
Complete_CnfMtrx_w  = cell(1,K);
Complete_CnfMtrx_wo  = cell(1,K);
NB_Predicteds = cell(1,K);

for k = 1:K
     fprintf('---------------------------- fold:%d ----------------------------\n',k);
     %% Partioning
     TestIndex = KFoldsIdx{k};
     TrainIndex = setdiff(AllIndex,TestIndex);
     Test_Normal_Dataset = DatasetN(TestIndex,:);
     Train_Dataset = DatasetN(TrainIndex,:);
   
     %% Train Anomaly Detector  
     [Train_Featureset,Scaling_Parameters_Pre] = Scale_Features_FFC(Train_Dataset(:,1:end-2),'balanced');
     Means = mean(Train_Featureset);
     SDs = std(Train_Featureset);
    
    %% Test 
    DatasetA = DatasetA_; 
    DatasetN_AD = Test_Normal_Dataset;
    DatasetN_AD(:,end-1) = 1;
    DatasetA(:,end-1) = 2;
    Test_Dataset = [DatasetN_AD ; DatasetA];
    Test_Featureset = Test_Dataset(:,1:end-2);
    [Test_Featureset,~] = Scale_Features_FFC(Test_Featureset,Scaling_Parameters_Pre);

    predictS = ones(1,size(Test_Featureset,1));
    for s = 1:size(Test_Featureset,1)
        sample = Test_Featureset(s,:);
        cnt = 0;
        for f=1:size(Test_Featureset,2)
            anomaly_cut_off = SDs(f)*3;
            upper_limit = Means(f) + anomaly_cut_off;
            lower_limit = Means(f) - anomaly_cut_off;
            if (sample(f) > upper_limit) || (sample(f) < lower_limit)
                cnt = cnt + 1;
                if cnt == 1
                    predictS(s) = 2; %anomaly
                    break
                end
            end
        end
    end
    AD_Predicted = predictS;
    Normal_Predicted_Index = find(AD_Predicted ==1); % This goes to Classifier
    Anomaly_Predicted_Index = find(AD_Predicted ==2);
    SS_Predicted{k} = AD_Predicted;
    
    DatasetA = DatasetA_; 
    DatasetN_AD = Test_Normal_Dataset;
    DatasetN_AD(:,end-1) = 1;
    DatasetA(:,end-1) = 2;
    Test_Dataset = [DatasetN_AD ; DatasetA];
    AD_TrueLabels = Test_Dataset(:,end-1);
    ADClassLabels = {'Normal','Anomaly'};
    Weights = Assign_Weights_FFC(Test_Dataset(:,end-1),ADClassLabels,'balanced');
    ADCnfMtrx1 = ConfusionMatrix_FFC(Test_Dataset(:,end-1),AD_Predicted',ADClassLabels,ADClassLabels,Weights);
    ADCnfMtrx{k} = Scale_ConfusionMatrix_FFC(ADCnfMtrx1);
    title = 'Anomaly Detector Confusion Matrix';
    ShowConfusionMatrix_FFC(ADCnfMtrx{k},ADClassLabels,ADClassLabels,title);
    
     %% Train Classifier
     DatasetCN = Train_Dataset;
     Weights = Assign_Weights_FFC(DatasetCN(:,end-1),NormalClassLabels,'balanced');
     MinLeafSize = .001;
     NumTrees = 100;
     TVIndex = [0 1];
     TVc = [.8 .2];
     TVc = TVc/sum(TVc);
     [~,TIndex,VIndex] = Partition_Dataset_FFC(DatasetCN(:,end-1:end),NormalClassLabels,...
        {[TVIndex(1) TVIndex(1)+(TVIndex(2)-TVIndex(1))*TVc(1)],...
        [TVIndex(1)+(TVIndex(2)-TVIndex(1))*TVc(1) TVIndex(2)]},PartitionGenerateError);
     [RandomForest_CL,Pc,ConfusionMatrix,Pc_Train,ConfusionMatrix_Train] = Build_RandomForest_FFC([],DatasetCN,NormalClassLabels,FeatureLabels,Weights,TIndex,VIndex,NumTrees,MinLeafSize);
     
    %% Test the Classifier (n+1 * n+1)
    AllClassLabels = [NormalClassLabels {'Anomaly'}];    
    DatasetA(:,end-1) = length(NormalClassLabels) + 1;
    Test_Dataset = [DatasetN(TestIndex,:) ; DatasetA];
    C_TrueLabels = Test_Dataset(:,end-1);
    Weights = Assign_Weights_FFC(Test_Dataset(:,end-1),AllClassLabels,'balanced');
    Test_Dataset = Test_Dataset(AD_Predicted==1,:);
    [~,~,~,C_Predicted,~] = Test_RandomForest_FFC(RandomForest_CL,Test_Dataset,(1:1:length(Test_Dataset)),NormalClassLabels,NormalClassLabels,FeatureLabels,FeatureLabels,Weights);
    C_PredictedLabels = ones(size(Test_Dataset,1),1);
    C_PredictedLabels(Anomaly_Predicted_Index) = length(NormalClassLabels)+1;
    C_PredictedLabels(Normal_Predicted_Index) = C_Predicted;
    Complete_CnfMtrx1 = ConfusionMatrix_FFC(C_TrueLabels,C_PredictedLabels,AllClassLabels,AllClassLabels,Weights);
    Complete_CnfMtrx_w{k} = Scale_ConfusionMatrix_FFC(Complete_CnfMtrx1);
    Accuracy = mean(diag(Complete_CnfMtrx_w{k}));
    title = 'Complete Confusion Matrix';
    ShowConfusionMatrix_FFC(Complete_CnfMtrx_w{k},AllClassLabels,AllClassLabels,title);
    fprintf('Accuracy when Anomaly Detector is used is: %f \n',Accuracy);
 
    %% Test Classifier w/o Anomaly Detector 
    AllClassLabels = [NormalClassLabels {'Anomaly'}];    
    DatasetA(:,end-1) = length(NormalClassLabels)+1;
    Test_Dataset = [DatasetN(TestIndex,:) ; DatasetA];
    C_TrueLabels = Test_Dataset(:,end-1);
    Weights = Assign_Weights_FFC(Test_Dataset(:,end-1),AllClassLabels,'balanced');
    [~,~,~,C_Predicted,~] = Test_RandomForest_FFC(RandomForest_CL,Test_Dataset,(1:1:length(Test_Dataset)),NormalClassLabels,NormalClassLabels,FeatureLabels,FeatureLabels,Weights);
    C_PredictedLabels = C_Predicted;
    Complete_CnfMtrx1 = ConfusionMatrix_FFC(C_TrueLabels,C_PredictedLabels,AllClassLabels,AllClassLabels,Weights);
    Complete_CnfMtrx_wo{k} = Scale_ConfusionMatrix_FFC(Complete_CnfMtrx1);
    Accuracy = mean(diag(Complete_CnfMtrx_wo{k}));
    title = 'Confusion Matrix without Anomaly Detector';
    ShowConfusionMatrix_FFC(Complete_CnfMtrx_wo{k},AllClassLabels,AllClassLabels,title);
    fprintf('Accuracy without Anomaly Detector is: %f \n',Accuracy);
end

% save SS_WS