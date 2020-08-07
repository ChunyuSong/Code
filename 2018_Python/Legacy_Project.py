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
from sklearn.preprocessing import StandardScaler

#%%time
print("Deep Neural Network for PCa grade classification: start...")

# project_dir = '/bmrp092temp/Zezhong_Ye/2018_Legacy_Project/machine_learning'
# results_dir = '/bmrp092temp/Zezhong_Ye/2018_Legacy_Project/machine_learning'

project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2018_Legacy_Project\machine_learning'
results_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2018_Legacy_Project\machine_learning'

# load data from excel files
print("loading data: start...")
df = pd.read_csv(os.path.join(project_dir, 'total_voxel_DNN.csv'))
# X = df.iloc[1:, 7:30]
# y = df.iloc[1:, [6]]
# x_list = ['X','Y','Z']
# test = df[x_list]
df.loc[df['ROI_Class'] == 'n', 'y_cat'] = 1
df.loc[df['ROI_Class'] == 't', 'y_cat'] = 2
df.loc[df['ROI_Class'] == 'nc', 'y_cat'] = 3
df.loc[df['ROI_Class'] == 'i', 'y_cat'] = 4
df.loc[df['ROI_Class'] == 'h', 'y_cat'] = 5
# df = df[df.y_cat != 5]
X = df.iloc[:, [8]]
y = df.y_cat
y = y.astype('int')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("data loading: complete!")
print("training set size:", len(X_train))
print("test set size:", len(X_test))

# training a SVM classifier
print("training a MLP classifier: start...")

# clf = SVC(C=1.0, cache_size=1000, class_weight='balanced', coef0=0.0, decision_function_shape=None,
#           degree=3, gamma=0.01, kernel='rbf', max_iter=-1, probability=False, random_state=None,
#           shrinking=True, tol=0.001, verbose=False)

#start = time.time()
clf = MLPClassifier(activation='tanh', hidden_layer_sizes=(100, 100, 100), alpha=0.0001, batch_size='auto',
                    beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, learning_rate='constant',
                    learning_rate_init=0.001, max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
                    random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
                    verbose=False, warm_start=False)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Deep Neural Network training: complete!")

print("calculating MS lesion types classification accuracy: start...")
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:", cm)

print("calculating precision, recall, f1-score: start...")
print(classification_report(y_test, y_pred, digits=3))

accuracy_class_1 = np.around(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]+cm[0,3]+cm[0,4]), 3)
accuracy_class_2 = np.around(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]+cm[1,3]+cm[1,4]), 3)
accuracy_class_3 = np.around(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]+cm[2,3]+cm[2,4]), 3)
accuracy_class_4 = np.around(cm[3,3]/(cm[3,0]+cm[3,1]+cm[3,2]+cm[3,3]+cm[3,4]), 3)
accuracy_class_5 = np.around(cm[4,4]/(cm[4,0]+cm[4,1]+cm[4,2]+cm[4,3]+cm[4,4]), 3)
# accuracy_class_1 = np.around(cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2]), 3)
# accuracy_class_2 = np.around(cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2]), 3)
# accuracy_class_3 = np.around(cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2]), 3)
accuracy_overall = np.around(accuracy, 2)

print("benign Accuracy:", accuracy_class_1)
print("tumor Accuracy:", accuracy_class_2)
print("necrosis Accuracy:", accuracy_class_3)
print("infiltration Accuracy:", accuracy_class_4)
print("hemorrhage Accuracy:", accuracy_class_5)


print("Tumor Classification Accuracy:", accuracy_overall)
print("calculating tumor classification accuracy: complete!")

### plot confusion matrix
print("plotting confusion matrix_1: start...")
ax_1 = sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap="Blues", linewidths=.5)
#plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)#for label size
#plt.ylabel('True label', fontsize=13, fontweight='bold')
#plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
plt.tight_layout()
ax_1.axhline(y=0, color='k', linewidth=3)
ax_1.axhline(y=5, color='k', linewidth=3)
ax_1.axvline(x=0, color='k', linewidth=3)
ax_1.axvline(x=5, color='k', linewidth=3)
ax_1.set_aspect('equal')
# plt.savefig(os.path.join(results_dir, 'confusion_matrix_1.png'), format='png', dpi=600)
# plt.show()
plt.close()
print("plotting confusion matrix_1: complete!")

# generate normalized confusion matrix
print("plotting confusion matrix_2: start...")
cm_2 = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_2 = np.around(cm_2, 2)
print(cm_2)
#ax_2 = fig.add_subplot(1,1,1)
ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
#plt.figure(figsize = (10,7))
#sn.set(font_scale=1.4)#for label size
#plt.ylabel('True label', fontsize=13, fontweight='bold')
#plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
plt.tight_layout()
ax_2.axhline(y=0, color='k', linewidth=3)
ax_2.axhline(y=5, color='k', linewidth=3)
ax_2.axvline(x=0, color='k', linewidth=3)
ax_2.axvline(x=5, color='k', linewidth=3)
ax_2.set_aspect('equal')
# plt.savefig(os.path.join(results_dir, 'confusion_matrix_2.png'), format='png', dpi=600)
# plt.show()
plt.close()
print("plotting confusion matrix_2: complete!")
print("MS lesion classification: successful!")
