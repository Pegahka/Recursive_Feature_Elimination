# Recursive_Feature_Elimination
Recursive Feature Elimination for MATLAB's fitcsvm (Matlab's Support Vector Machine Implementation)

This code combines feature selection and feature ranking with SVM-based classification the respective data.

This code combines Rescursive Feature Elimination (RFE) as described in Gene Selection for Cancer 
Classification using Support Vector Machines, Guyon et al. 2002 with MATLAB's Support Vector Machine (SVM) Implementation **fitcsvm**.

The final output of this approach delivers a ranking of the features through classification-based
feature selection. 

The code is based on KE YAN's code where RFE was combined with the older Matlab SVM implementation (svmtrain):
https://ch.mathworks.com/matlabcentral/fileexchange/50701-feature-selection-with-svm-rfe

