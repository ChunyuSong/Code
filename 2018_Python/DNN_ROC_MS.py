from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import glob2 as glob
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


print("PCa DNN classification ROC analysis: start...")

project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI'
results_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI\results'

# load data from excel files
print("loading data: start...")
df = pd.read_csv(os.path.join(project_dir, 'MS_test_1256.csv'), header=None)
X = df.iloc[1:, 13:24]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype(np.float64))
y1 = df.iloc[1:, 25].astype('int')
y = label_binarize(y1, classes=sorted(y1.unique()))
n_classes = 4


print("data loading: complete!")
print("training a DNN classifier: start...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("training set size:", len(X_train))
print("test set size:", len(X_test))

clf = OneVsRestClassifier(MLPClassifier(activation='tanh', hidden_layer_sizes=(300, 300, 300), alpha=0.0001,
                                        batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
                                        epsilon=1e-08, learning_rate='constant', learning_rate_init=0.001,
                                        max_iter=500, momentum=0.9, nesterovs_momentum=True, power_t=0.5,
                                        random_state=None, shuffle=True, solver='adam', tol=0.0001,
                                        validation_fraction=0.1, verbose=False, warm_start=False))
clf.fit(X_train, y_train)
print("training a DNN classifier: complete!")

y_score = clf.predict_proba(X_test)
#y_pred = clf.predict(X_test)

### Compute ROC curve and AUC for each class
print("plotting ROC curves: start...")
fpr = dict()
tpr = dict()
threshold = dict()
roc_auc = dict()

for i in range(n_classes):
    #fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(np.around(roc_auc[i], 3))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #plt.title('ROC', fontsize=14, fontweight='bold')
    plt.plot(fpr[i], tpr[i], color='royalblue', linewidth=5, label='AUC = %0.3f'%roc_auc[i])
    plt.legend(loc='lower right')
    plt.legend(fontsize=24)
    legend_properties = {'weight': 'bold'}
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlim([-0.02, 1])
    plt.ylim([0, 1.02])
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=20)
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=20)
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.02, color='k', linewidth=4)
    ax.axvline(x=-0.02, color='k', linewidth=4)
    ax.axvline(x=1, color='k', linewidth=4)
    plt.grid(True)
    ax.set_aspect('equal')
    #plt.show()
    #plt.savefig(os.path.join(results_dir, 'ROC_'+str(x)+'.png', format='png', dpi=600)
    plt.close()

print("plotting ROC curves: complete!")
print("PCa SVM classification ROC analysis: complete!")