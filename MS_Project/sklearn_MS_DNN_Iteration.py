

import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

start = time.time()
print("Deep Neural Network for PCa grade classification: start...")

### MLP paramaters
n = 5 # times of iteration
test_size = 0.2
random_state = 42
hidden_layer_sizes = (256, 256, 256, 256, 256)
batch_size = 200
learning_rate_init = 0.001
momentum = 0.9
d = 3 # decimal numbers
activation = 'tanh' # activation function: tanh, relu
solver = 'adam' # optimized function

print("loading data: start...")
project_dir = r'\\10.39.42.102\temp\2019_MS\AI'
results_dir = r'\\10.39.42.102\temp\2019_MS\AI\results'

#complete list of result maps
maps_list = [
             'dti_fa_map.nii',           #15
             'dti_adc_map.nii',          #16
             'dti_axial_map.nii',        #17
             'dti_radial_map.nii',       #18
             'fiber_ratio_map.nii',      #19
             'fiber1_fa_map.nii',        #20
             'fiber1_axial_map.nii',     #21
             'fiber1_radial_map.nii',    #22
             'restricted_ratio_map.nii', #23
             'hindered_ratio_map.nii',   #24
             'water_ratio_map.nii',      #25
             'b0_map.nii',               #26
             'T2W',                      #27
             'FLAIR',                    #28
             'MPRAGE',                   #29
             'MTC'                       #30
]

df = pd.read_csv(os.path.join(project_dir, '20190302.csv'))
X = df.iloc[:, [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]]
y = df.iloc[:, [31]].astype('int')

accuracy = list()
accuracy_class_1 = list()
accuracy_class_2 = list()
accuracy_class_3 = list()
accuracy_class_4 = list()
accuracy_class_5 = list()
cm_percentage = np.zeros(shape=(5, 5, n))
cm_count = np.zeros(shape=(5, 5, n))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))

X_train, X_test, y_train, y_test = train_test_split(
                                                    X_scaled,
                                                    y,
                                                    test_size=test_size,
                                                    random_state=random_state
)

print("data loading: complete!")
print("training set size:", len(X_train))
print("test set size:", len(X_test))
print("training a MLP classifier: start...")

for i in range(n):

    clf = MLPClassifier(
                        activation=activation,
                        hidden_layer_sizes=hidden_layer_sizes,
                        alpha=0.0001,
                        batch_size=batch_size,
                        beta_1=0.9,
                        beta_2=0.999,
                        early_stopping=False,
                        epsilon=1e-08,
                        learning_rate='constant',
                        learning_rate_init=learning_rate_init,
                        max_iter=500,
                        momentum=momentum,
                        nesterovs_momentum=True,
                        power_t=0.5,
                        random_state=None,
                        shuffle=True,
                        solver=solver,
                        tol=0.0001,
                        validation_fraction=0.1,
                        verbose=False,
                        warm_start=False
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy.append(np.round(accuracy_score(y_test, y_pred), d))

    print("Deep Neural Network training %s: complete!" %(i+1))
    print("confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("normalized confusion matrix:")
    cm_2 = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    cm_2 = np.around(cm_2, 2)
    print(cm_2)
    
    print("precision, recall, f1-score report:")
    print(classification_report(y_test, y_pred, digits=d))
    
    accuracy_class_1.append(np.around(cm_2[0,0]/(cm_2[0,0]+cm_2[0,1]+cm_2[0,2]+cm_2[0,3]+cm_2[0,4]), d))
    accuracy_class_2.append(np.around(cm_2[1,1]/(cm_2[1,0]+cm_2[1,1]+cm_2[1,2]+cm_2[1,3]+cm_2[1,4]), d))
    accuracy_class_3.append(np.around(cm_2[2,2]/(cm_2[2,0]+cm_2[2,1]+cm_2[2,2]+cm_2[2,3]+cm_2[2,4]), d))
    accuracy_class_4.append(np.around(cm_2[3,3]/(cm_2[3,0]+cm_2[3,1]+cm_2[3,2]+cm_2[3,3]+cm_2[3,4]), d))
    accuracy_class_5.append(np.around(cm_2[4,4]/(cm_2[4,0]+cm_2[4,1]+cm_2[4,2]+cm_2[4,3]+cm_2[4,4]), d))

    cm_count[:, :, i] = cm
    cm_percentage[:, :, i] = cm_2

print('PBH Accuracy:', accuracy_class_1, 'Mean Accuracy:', np.around(np.mean(accuracy_class_1), d))
print('PGH Accuracy:', accuracy_class_2, 'Mean Accuracy:', np.around(np.mean(accuracy_class_2), d))
print('AGH Accuracy:', accuracy_class_3, 'Mean Accuracy:', np.around(np.mean(accuracy_class_3), d))
print('T2W Accuracy:', accuracy_class_4, 'Mean Accuracy:', np.around(np.mean(accuracy_class_4), d))
print('NAWM Accuracy:', accuracy_class_5, 'Mean Accuracy:', np.around(np.mean(accuracy_class_5), d))
print('Overall Accuracy:', accuracy, 'Mean Accuracy:', np.around(np.mean(accuracy), d))
print("calculating MS lesion classification accuracy: complete!")
    
### plot confusion matrix
avg_cm_cnt = np.mean(cm_count, axis=2)
print("plotting confusion matrix_1: start...")
ax_1 = sn.heatmap(avg_cm_cnt, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)#fmt='d', 
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
plt.savefig(os.path.join(results_dir, 'confusion_matrix_1.png'), format='png', dpi=600)
# plt.show()
plt.close()
print("plotting confusion matrix_1: complete!")

#ax_2 = fig.add_subplot(1,1,1)
avg_cm_per = np.mean(cm_percentage, axis=2)
ax_2 = sn.heatmap(avg_cm_per, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
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
plt.savefig(os.path.join(results_dir, 'confusion_matrix_2.png'), format='png', dpi=600)
# plt.show()
plt.close()
print("plotting confusion matrix_2: complete!")
print("MS lesion classification: successful!")
print(time.time(), time.clock())