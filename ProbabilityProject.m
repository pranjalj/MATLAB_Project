%Bayes Classification
%Pranjal Jain
%Tanya Jain

%loading data 
load('data.mat')

%Step 1.Training: Use the data of the first 100 subjects

train_f1 = F1(1:100,:);
test_f1 = F1(101:1000,:);
train_f2 = F2(1:100,:);
test_f2 = F2(101:1000,:);

% Step 2.1.Testing: Assume that X = F1. Using the Bayes theorem,
% calculate the probability of each class for data of the remaining 
% subjects (columns 101-1000 of F1) 
% and consequently predict the class for each data point. 
% Note that each subject performed 5 different tasks 
% so you need to predict the class of 4500 data points.

%creating matrix to refrence class
original_class = zeros(900, 5);
for ind = 1:5
    original_class(:,ind) = ind;
end

%estimate mean and variance for f1
mean_f1 = mean(train_f1);
var_f1 = var(train_f1);

%predicting class of every element of test_f1
predicting_class_f1 = predict(test_f1,mean_f1,var_f1);

% Step 2.2.Calculating the accuracy of the classifier: 
% You need to check the percentage of the data whose class are 
% correctly predicted. 
% The true class is the row number of the data.

%initilaize accurate
accurate_f1 = 0;

%find number of element correctly classified
for ind = 1:5
    accurate_f1 = accurate_f1 + ...
        sum((original_class(:,ind) - predicting_class_f1(:,ind) == 0));
end

%classification_accuracy
classification_accuracy_f1 = accurate_f1/4500;
%error rate
error_rate_f1 = (4500-accurate_f1)/4500;

% Step 3. Standard Normal (Z-Score): 
% In this case the mean value and the range of data reported by one subject 
% will not be consistent with another subject. 
% In other to remove the effect of individual differences, 
% you have to normalize the data of each subject using 
% the standard normal formulation 
% (removing the mean and dividing by standard deviation). 
% Calculate Z1 (the standard normal of F1) and plot 
% the distribution of the data using Z1 and F2, 
% and compare it to the distribution in F1 and F2 shown on right.

%initializing Z1
Z1 = zeros(1000, 5);

%normalize value of every element of test_f1
for ind = 1:1000
    Z1(ind,:) = zscore(F1(ind,:));
end

% Plotting Z1 and F2 to derive inferences of their correlation
%F1 and F2
figure;
for ind = 1:5
    scatter(Z1(:,ind),F2(:,ind))
    hold on
end

title('Z1 vs F2')
xlabel('1^{st} Feature (Z1)')
ylabel('2^{nd} Feature (F2)')
legend('C1','C2','C3','C4','C5')
hold off

% Plotting F1 and F2 to derive inferences of their correlation
%Z1 and Z2
figure;
for i = 1:5
    scatter(F1(:,i),F2(:,i))
    hold on
end
title('F1 vs F2')
xlabel('1^{st} Feature (F1)')
ylabel('2^{nd} Feature (F2)')
legend('C1','C2','C3','C4','C5')
hold off

% Step 4. Repeat 2.1 and 2.2 for the following cases:
%Case 2: X = Z1 (Note for this case you need to repeat 
%                   the training step as well)
%Case 3: X =F2
%Case 4: X = [Z1--F2]. Note that this is a multivariate normal distribution 
%                       and you need to use the independence assumption.

%CASE 2

%Training: Use the data of the first 100 subjects

train_z1 = Z1(1:100,:);
test_z1 = Z1(101:1000,:);

%estimate mean and variance for z1
mean_z1 = mean(train_z1);
var_z1 = var(train_z1);

%predicting class of every element of test_z1
predicting_class_z1 = predict(test_z1,mean_z1,var_z1);

%initilaize accurate
accurate_z1 = 0;

%find number of element correctly classified
for ind = 1:5
    accurate_z1 = accurate_z1 + ...
        sum((original_class(:,ind) - predicting_class_z1(:,ind) == 0));
end

%classification_accuracy ratio
classification_accuracy_z1 = accurate_z1/4500;
%error rate
error_rate_z1 = (4500-accurate_z1)/4500;

%CASE 3

%estimate mean and variance for f2
mean_f2 = mean(train_f2);
var_f2 = var(train_f2);

%predicting class of every element of test_z1
predicting_class_f2 = predict(test_f2,mean_f2,var_f2);

%initilaize accurate
accurate_f2 = 0;

%find number of element correctly classified
for ind = 1:5
    accurate_f2 = accurate_f2 + ...
        sum((original_class(:,ind) - predicting_class_f2(:,ind) == 0));
end

%classification_accuracy ratio
classification_accuracy_f2 = accurate_f2/4500;
%error rate
error_rate_f2 = (4500-accurate_f2)/4500;

%CASE 4

%predicting class of every element of test_z1
predicting_class_multivariant = ...
                        predict_multivariant(test_z1,test_f2,...
                            mean_z1,mean_f2,var_z1,var_f2);

%initilaize accurate
accurate_multivariant = 0;

%find number of element correctly classified
for ind = 1:5
    accurate_multivariant = accurate_multivariant + ...
        sum((original_class(:,ind) - ...
        predicting_class_multivariant(:,ind) == 0));
end

%classification_accuracy ratio
classification_accuracy_multivariant = accurate_multivariant/4500;
%error rate
error_rate_multivariant = (4500-accurate_multivariant)/4500;

%Step 5. Compare the classification rate of the four cases.

%assigning classification accuracy values
classification_accuracy(1) = classification_accuracy_f1;
classification_accuracy(2) = classification_accuracy_z1;
classification_accuracy(3) = classification_accuracy_f2;
classification_accuracy(4) = classification_accuracy_multivariant;

figure;
x = (categorical({'F1','Z1','F2','Z1--F2'}));
y = classification_accuracy;
bar(x,y)
for i1=1:numel(y)
    text(x(i1),y(i1)/2,num2str(y(i1),'%0.2f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
title('Classification rate of different classes')
ylabel('Classification Accuracy')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Defining Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [predicting_class] = predict(data,mean,var)
%predict fuction takes dataset, mean and variance and return prected value

%initalizing predicted class matrix
predicting_class = zeros(900, 5);

%initalizng matrix to store z score temporarily
norm_temp = zeros(5,1);

for ind1 = 1:900
    for ind2 = 1:5
        for ind_temp = 1:5
        %finding z_score for every element for considering it of every
        %class
        norm_temp(ind_temp,1) = (data(ind1,ind2) - ...
                                    mean(ind_temp))/(sqrt(var(ind_temp)));
        end
      %finding max to all z-score tp predict class of value
      [~,predicting_class(ind1,ind2)] = max(normpdf(norm_temp)); 
    end
end
end

function [predicting_class] = predict_multivariant(data1,data2,mean1, ...
                                    mean2,var1,var2)
%predict fuction takes dataset, mean and variance and return prected value

%initalizing predicted class matrix
predicting_class = zeros(900, 5);

%initalizng matrix to store z score temporarily
norm_temp = zeros(5,2);

for ind1 = 1:900
    for ind2 = 1:5
        for ind_temp = 1:5
        %finding z_score for every element for considering it of every
        %class
        norm_temp(ind_temp,1) = (data1(ind1,ind2) - ...
                                   mean1(ind_temp))/(sqrt(var1(ind_temp)));
        norm_temp(ind_temp,2) = (data2(ind1,ind2) - ...
                                   mean2(ind_temp))/(sqrt(var2(ind_temp)));
        end
        norm_temp(:,1) = normpdf(norm_temp(:,1));
        norm_temp(:,2) = normpdf(norm_temp(:,2));
        %finding max to all z-score tp predict class of value
        [~,predicting_class(ind1,ind2)] = ...
                                max(norm_temp(:,1).*norm_temp(:,2)); 
    end
end
end