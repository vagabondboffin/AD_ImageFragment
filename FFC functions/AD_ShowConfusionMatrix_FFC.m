function AD_ShowConfusionMatrix_FFC(Complete_CnfMtrx,ADCnfMtrx,NormalClassLabels)
K = length(Complete_CnfMtrx); %Fold
AllClassLabels = [NormalClassLabels {'Anomaly'}];    

for j = 1:K
    mainTitle = sprintf('--- %s%d%s ---','Confusion Matrix for ',j,'th fold');
    
    fprintf(mainTitle);
    
    title = 'Anomaly Detector Confusion Matrix';
    ShowConfusionMatrix_FFC(ADCnfMtrx{j},{'Normal' 'Anomaly'},{'Normal' 'Anomaly'},title);
    
    title = 'Complete Confusion Matrix';
    ShowConfusionMatrix_FFC(Complete_CnfMtrx{j},AllClassLabels,AllClassLabels,title);    
    
end
