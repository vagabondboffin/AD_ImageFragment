currentFolder = which('ScriptForAE.m');
currentFolder(strfind(currentFolder,'ScriptForAE.m')-1:end) = [];
AllPaths_FFC = genpath(currentFolder);
addpath(AllPaths_FFC);

load('normalDatasetPath')
DatasetN = Dataset;
NormalClassLabels = ClassLabels;

load('anomalyDatasetPath');
DatasetA_ = Dataset;

load('Autoencoders.mat');

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
for k = 1:K
    fprintf('---------------------------- fold:%d ----------------------------\n',k);
     %% Partioning
     TestIndex = KFoldsIdx{k};
     TrainIndex = setdiff(AllIndex,TestIndex);
     Test_Normal_Dataset = DatasetN(TestIndex,:);
     Train_Dataset = DatasetN(TrainIndex,:);
     
    % **********************************************************************
    % *******************************  Train ******************************* 
    % **********************************************************************
    %% Train Anomaly Detector for Other Image Formats 
    %% Scaling (pre)
    [Train_Featureset,Scaling_Parameters_Pre] = Scale_Features_FFC(Train_Dataset(:,1:end-2),'z-score');
    
    %% Feature Selection and Encoding Train set
    
    Autoenc1 = Autoencoders{1,k};
    Train_Featureset1 = encode(Autoenc1,Train_Featureset');

    Autoenc2 = Autoencoders{2,k};
    Train_Featureset2 = encode(Autoenc2,Train_Featureset1);

    Autoenc3 = Autoencoders{3,k};
    Train_Featureset3 = encode(Autoenc3,Train_Featureset2);

    Autoenc4 = Autoencoders{4,k};
    Train_Featureset4 = encode(Autoenc4,Train_Featureset3);

    y1 = decode(Autoenc4, Train_Featureset4);
    y2 = decode(Autoenc3, y1);
    y3 = decode(Autoenc2, y2);
    y4 = decode(Autoenc1, y3);

    y_hat = y4;
    % Calculate the error
    Train_Err = sqrt(sum((y_hat - Train_Featureset').^2)); 
    Train_Errs{k} = Train_Err;
    Train_Err = round(Train_Err);
    Thrshld = mean(Train_Err);
    
    % **********************************************************************
    % *******************************  Test  ******************************* 
    % **********************************************************************
    %% Test 
    DatasetA = DatasetA_; 
    DatasetN_AD = Test_Normal_Dataset;
    DatasetN_AD(:,end-1) = 1;
    DatasetA(:,end-1) = 2;
    Test_Dataset = [DatasetN_AD ; DatasetA];
   
    [Test_Featureset,~] = Scale_Features_FFC(Test_Dataset(:,1:end-2),Scaling_Parameters_Pre);
    
    Test_Featureset1 = encode(Autoenc1,Test_Featureset');
    Test_Featureset2 = encode(Autoenc2,Test_Featureset1);
    Test_Featureset3 = encode(Autoenc3,Test_Featureset2);
    Test_Featureset4 = encode(Autoenc4,Test_Featureset3);

    y1 = decode(Autoenc4, Test_Featureset4);
    y2 = decode(Autoenc3, y1);
    y3 = decode(Autoenc2, y2);
    y4 = decode(Autoenc1, y3);

    y_hat = y4;
    % Calculate the error
    Test_Errs = sqrt(sum((y_hat - Test_Featureset').^2)); 

    AD_Predicted = ones(1,size(Test_Dataset,1));
    AD_Predicted(Test_Errs > Thrshld) = 2;    
    AD_Predicteds{k} = AD_Predicted;
    Normal_Predicted_Index = find(AD_Predicted ==1); % This goes to Classifier
    Anomaly_Predicted_Index = find(AD_Predicted ==2);
    
    ADClassLabels = {'Normal','Anomaly'};
    Weights = Assign_Weights_FFC(Test_Dataset(:,end-1),ADClassLabels,'balanced');
    ADCnfMtrx1 = ConfusionMatrix_FFC(Test_Dataset(:,end-1),AD_Predicted',ADClassLabels,ADClassLabels,Weights);
    ADCnfMtrx{k} = Scale_ConfusionMatrix_FFC(ADCnfMtrx1);
    title = 'Anomaly Detector Confusion Matrix';
    ShowConfusionMatrix_FFC(ADCnfMtrx{k},ADClassLabels,ADClassLabels,title);
    
     % **********************************************************************
     % ****************************  Classifier  ****************************
     % **********************************************************************
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
    DatasetA(:,end-1) = length(NormalClassLabels)+1;
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
