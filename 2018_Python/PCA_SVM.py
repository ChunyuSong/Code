# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import glob2 as glob
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import string
from matplotlib.pylab import *


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

# split data into train and test dataset as 4:1
train, test = train_test_split(df, test_size=0.2)

# X -> DHisto and DTI features, y -> labels
X_train = train.iloc[:, [1, 2, 3, 8, 11, 14, 18, 19, 20, 22, 26, 27, 28, 29, 32, 37, 38, 40]]
y_train = train.iloc[:, 0]

X_test = test.iloc[:, [1, 2, 3, 8, 11, 14, 18, 19, 20, 22, 26, 27, 28, 29, 32, 37, 38, 40]]
y_test = test.iloc[:, 0]

# training a SVM classifier
svm_model = SVC(C=1500.0, cache_size=1000, class_weight='balanced', coef0=0.0, decision_function_shape=None, degree=5,
                gamma=0.5, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001,
                verbose=False).fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

# model accuracy for X_test
accuracy = svm_model.score(X_test, y_test)
print(confusion_matrix(y_test, svm_predictions))

### calculate individual class accuracy
cm = confusion_matrix(y_test, svm_predictions)
accuracy_class_1 = cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4])
accuracy_class_2 = cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4])
accuracy_class_3 = cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4])
accuracy_class_4 = cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4])
accuracy_class_5 = cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4])
overall_accuracy = accuracy

print(accuracy_class_1)
print(accuracy_class_2)
print(accuracy_class_3)
print(accuracy_class_4)
print(accuracy_class_5)
print(overall_accuracy)

### plot confusion matrix heat map

ax = sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap="Blues", linewidths=.5)
#plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)#for label size
#plt.ylabel('True label', fontsize=13, fontweight='bold')
#plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
plt.tight_layout()
ax.axhline(y=0, color='k',linewidth=3)
ax.axhline(y=5, color='k',linewidth=3)
ax.axvline(x=0, color='k',linewidth=3)
ax.axvline(x=5, color='k',linewidth=3)
ax.set_aspect('equal')
plt.show()
