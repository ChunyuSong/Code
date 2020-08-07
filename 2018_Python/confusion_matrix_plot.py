import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import string
from matplotlib.pylab import *


array = [[0.538, 0.462, 0],
         [0.718, 0.282, 0],
         [0.03, 0.967, 0]]
#df_cm = pd.DataFrame(array, range(6),
                  # range(6))
#df_cm = pd.DataFrame(array, index=["Hypercellular Tumor", "Tumor Necrosis", "Tumor Infiltration"],
                     #columns=["Hypercellular Tumor", "Tumor Necrosis", "Tumor Infiltration"])
#ax = sn.heatmap(array, annot=True, annot_kws={"size": 16}, fmt="d", cmap="Blues", linewidths=.8)
ax = sn.heatmap(array, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.8)
# plt.figure(figsize = (10,7))
# sn.set(font_scale=1.4)#for label size
#plt.ylabel('True label', fontsize=13, fontweight='bold')
#plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
plt.tight_layout()
ax.axhline(y=0, color='k',linewidth=3)
ax.axhline(y=3, color='k',linewidth=3)
ax.axvline(x=0, color='k',linewidth=3)
ax.axvline(x=3, color='k',linewidth=3)
ax.set_aspect('equal')
plt.show()
