import pandas as pd
import glob2 as glob
from sklearn.svm import SVC
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import string
from matplotlib.pylab import *
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


print("Deep Neural Network for PCa grade classification: start...")

project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCA_in_vivo_data_excel'
results_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning'

# project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCA_NCCN_Risk\data'
# results_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCA_NCCN_Risk'


# load data from excel files
print("loading data: start...")
files = glob.glob(project_dir + "/*.xlsx")
df = pd.DataFrame()
for f in files:
    data = pd.read_excel(f, 'Sheet1', header=None)
    data.iloc[:, 1:] = (data.iloc[:, 1:])/(data.iloc[:, 1:].max())
    data = data.iloc[data[data.iloc[:, 0] != 0].index]
    df = df.append(data)

# split data into train and test dataset as 4:1
#X = df.iloc[:, [1, 2, 3, 8, 11, 14, 18, 19, 20, 22, 26, 27, 28, 29, 32, 37, 38, 40]] ##
#X = df.iloc[:, [1,14,18,19,20,22,26,27,28,29,31,32,33,35,36,37,38,39,40]]  #DBSI dataset
#X = df.iloc[:, [2,3,8,11]] #DTI dataset
#X = df.iloc[:, [2]] #ADC dataset
X = df.iloc[:, [1,2,3,8,11,13,14,18,19,20,22,26,27,28,29,31,32,33,35,36,37,38,39,40]]  #whole dataset

y = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("data loading: complete!")
print("training set size:", len(X_train))
print("test set size:", len(X_test))

# training a SVM classifier
print("training a MLP classifier: start...")

start = time.time()
clf = MLPClassifier(activation='tanh',
                    hidden_layer_sizes=(100,100,100),
                    alpha=0.0001,
                    batch_size='auto',
                    beta_1=0.9,
                    beta_2=0.999,
                    early_stopping=False,
                    epsilon=1e-08,
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_iter=500,
                    momentum=0.9,
                    nesterovs_momentum=True,
                    power_t=0.5,
                    random_state=None,
                    shuffle=True,
                    solver='adam',
                    tol=0.0001,
                    validation_fraction=0.1,
                    verbose=False,
                    warm_start=False).fit(X_train, y_train)

## test dataset
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Deep Neural Network training: complete!")

print("calculating PCa classification accuracy: start...")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("calculating precision, recall, f1-score: start...")
print(classification_report(y_test, y_pred, digits=3))

accuracy_class_1 = np.around(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]), 3)
accuracy_class_2 = np.around(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]), 3)
accuracy_class_3 = np.around(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]), 3)
accuracy_class_4 = np.around(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]), 3)
accuracy_class_5 = np.around(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]), 3)
accuracy_overall = np.around(accuracy, 2)

print('PCa Grade 1 Accuracy:', accuracy_class_1)
print('PCa Grade 2 Accuracy:', accuracy_class_2)
print('PCa Grade 3 Accuracy:', accuracy_class_3)
print('PCa Grade 4 Accuracy:', accuracy_class_4)
print('PCa Grade 5 Accuracy:', accuracy_class_5)
print('PCa Classification Accuracy:', accuracy_overall)
print("calculating PCa classification accuracy: complete!")

### plot confusion matrix
print("plotting confusion matrix_1: start...")
ax_1 = sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap="Blues", linewidths=.5)
#plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)#for label size
#plt.ylabel('True label', fontsize=13, fontweight='bold')
#plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
plt.tight_layout()
ax_1.axhline(y=0, color='k',linewidth=3)
ax_1.axhline(y=5, color='k',linewidth=3)
ax_1.axvline(x=0, color='k',linewidth=3)
ax_1.axvline(x=5, color='k',linewidth=3)
ax_1.set_aspect('equal')
plt.savefig(os.path.join(results_dir,'confusion_matrix_1.png'),format='png',dpi=600)
plt.show()
plt.close()
print("plotting confusion matrix_1: complete!")

# generate normalized confusion matrix
print("plotting confusion matrix_2: start...")
cm_2 = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_2 = np.around(cm_2,2)
print(cm_2)
#ax_2 = fig.add_subplot(1,1,1)
ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
#plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)#for label size
#plt.ylabel('True label', fontsize=13, fontweight='bold')
#plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
plt.tight_layout()
ax_2.axhline(y=0, color='k',linewidth=3)
ax_2.axhline(y=5, color='k',linewidth=3)
ax_2.axvline(x=0, color='k',linewidth=3)
ax_2.axvline(x=5, color='k',linewidth=3)
ax_2.set_aspect('equal')
plt.savefig(os.path.join(results_dir,'confusion_matrix_2.png'),format='png',dpi=600)
plt.show()
plt.close()
print("plotting confusion matrix_2: complete!")
print("PCa cancer grade classification: successful!")
