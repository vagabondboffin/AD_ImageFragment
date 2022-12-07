function [LastRow , LastColumn] = AD_ConfusionMatrix_FFC(AD_TrueLabels,AD_Predicted,C_TrueLabels,C_Predicted)
j = length(unique(C_TrueLabels)) - 1; %10
c = length(unique(C_TrueLabels)); %11
m = length(C_TrueLabels); % number of all test samples
Anomaly_Predicted_Index = find(AD_Predicted == 2);
LastRow = zeros(1,j); %1 ta 11
LastColumn = zeros(c,1);
for i = 1:m
    if isempty(find(Anomaly_Predicted_Index == i , 1)) % it's predicted as normal
        if(AD_TrueLabels(i) == 2)  % but it's anomaly
            Class = C_Predicted(i);
            LastRow(1,Class) = LastRow(Class) + 1;
        end
    elseif ~isempty(find(Anomaly_Predicted_Index == i , 1)) % it's predicted as anomaly
        if(AD_TrueLabels(i) == 1)
            Class = C_TrueLabels(i);
            LastColumn(Class) = LastColumn(Class) + 1;
        elseif(AD_TrueLabels(i) == 2)
            LastColumn(c) = LastColumn(c) + 1;
        end
    end
end

        
end