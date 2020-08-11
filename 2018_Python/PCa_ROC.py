# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import glob2 as glob
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt

# load data from excel files
#path = r'C:\Users\csong01\Desktop\PCA_in_vivo_data_excel_8'
path = r'/bmrp092temp/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCA_in_vivo_data_excel'

filenames = glob.glob(path + "/*.xlsx")

df = pd.DataFrame()
for f in filenames:
    data = pd.read_excel(f,'Sheet1',header = None)
    data.iloc[:, 1:] = (data.iloc[:, 1:]) / (data.iloc[:, 1:].max())
    data = data.iloc[data[data.iloc[:, 0] != 0].index]
    df = df.append(data)

X = df.iloc[:,[1,2,3,8,11,14,18,19,20,22,26,27,28,29,32,37,38,40]]
y1 = df.iloc[:,0]
y = label_binarize(y1, classes = sorted(y1.unique()))
n_classes = 9

# split data into train and test dataset as 4:1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# training a SVM classifier
svm_model = OneVsRestClassifier(SVC(C=1000.0, cache_size=1000, class_weight='balanced', coef0=0.0, decision_function_shape=None, degree=5,
                gamma=0.5, kernel='poly', max_iter=-1, probability=False, random_state=0, shrinking=True, tol=0.001,
                verbose=False)).fit(X_train, y_train)

y_score = svm_model.decision_function(X_test)
y_predict = svm_model.predict(X_test)

# model accuracy for X_test
accuracy = svm_model.score(X_test, y_test)
matrix = confusion_matrix(y_test, y_predict)
print(confusion_matrix(y_test, y_predict))

# Compute ROC curve and AUC for each class
import matplotlib.pyplot as plt
fpr = dict()
tpr = dict()
threshold = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr[i], tpr[i], 'b', label = 'AUC = %0.2f' % roc_auc[i])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()




