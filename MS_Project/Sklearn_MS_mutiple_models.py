import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sn
import string
from matplotlib.pylab import *
import os

from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



print("loading data: start...")

project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI'
results_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI\results'

df = pd.read_csv(os.path.join(project_dir, '20190302.csv'))
                
x = df.iloc[:, [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]].values
y = df.iloc[:, [31]].astype('int').values
                 
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x.astype(np.float64))

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.1, random_state=42)

##model = SVC(
##            C=100.0,
##            cache_size=1000,
##            class_weight='balanced',
##            coef0=0.0,
##            decision_function_shape=None,
##            degree=3,
##            gamma=0.5,
##            kernel='rbf',
##            max_iter=500,
##            probability=False,
##            random_state=None,
##            shrinking=True,
##            tol=0.001,
##            verbose=False
##            ).fit(x_train, y_train)

##clf = MLPClassifier(activation='relu', hidden_layer_sizes=(300, 300, 300), alpha=0.0001, batch_size='auto',
##                    beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, learning_rate='constant',
##                    learning_rate_init=0.001, max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
##                    random_state=None, shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
##                    verbose=False, warm_start=False)

model = LogisticRegression(
                            penalty='l2',
                            dual=False,
                            tol=0.0001,
                            C=1.0,
                            fit_intercept=True,
                            intercept_scaling=1,
                            class_weight=None,
                            random_state=None,
                            solver='lbfgs',
                            max_iter=500,
                            multi_class='auto',
                            verbose=0,
                            warm_start=False,
                            n_jobs=None,
                            ).fit(x_train, y_train)

##model = RandomForestClassifier(max_depth=10, random_state=0).fit(x_train, y_train)
        
y_pred = model.predict(x_test)

test_acc = accuracy_score(y_test, y_pred)
test_acc = np.around(test_acc, 3)
error_rate = np.around(1-test_acc, 3)

print('test accuracy:', test_acc)
print('error rate:', error_rate)

print('confusion matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("normalized confusion matrix:")
cm_2 = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_2 = np.around(cm_2, 2)
print(cm_2)

print("calculating precision, recall, f1-score: start...")
print(classification_report(y_test, y_pred, digits=3))

print("training set size:", len(x_train))
print("test set size:", len(x_test))

##### plot confusion matrix
##print("plotting confusion matrix_1: start...")
##ax_1 = sn.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap="Blues", linewidths=.5)
###plt.figure(figsize = (10,7))
###sn.set(font_scale=1.4)#for label size
###plt.ylabel('True label', fontsize=13, fontweight='bold')
###plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
##plt.tight_layout()
##ax_1.axhline(y=0, color='k',linewidth=3)
##ax_1.axhline(y=4, color='k',linewidth=3)
##ax_1.axvline(x=0, color='k',linewidth=3)
##ax_1.axvline(x=4, color='k',linewidth=3)
##ax_1.set_aspect('equal')
##plt.savefig(os.path.join(results_dir,'confusion_matrix_1.png'),format='png',dpi=600)
### plt.show()
##plt.close()
##print("plotting confusion matrix_1: complete!")

### generate normalized confusion matrix
##print("plotting confusion matrix_2: start...")
##cm_2 = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
##cm_2 = np.around(cm_2, 2)
##print(cm_2)
###ax_2 = fig.add_subplot(1,1,1)
##ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
###plt.figure(figsize = (10,7))
###sn.set(font_scale=1.4)#for label size
###plt.ylabel('True label', fontsize=13, fontweight='bold')
###plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
##plt.tight_layout()
##ax_2.axhline(y=0, color='k',linewidth=3)
##ax_2.axhline(y=4, color='k',linewidth=3)
##ax_2.axvline(x=0, color='k',linewidth=3)
##ax_2.axvline(x=4, color='k',linewidth=3)
##ax_2.set_aspect('equal')
##plt.savefig(os.path.join(results_dir,'confusion_matrix_2.png'),format='png',dpi=600)
### plt.show()
##plt.close()
##print("plotting confusion matrix_2: complete!")
##print("MS lesion classification: successful!")
