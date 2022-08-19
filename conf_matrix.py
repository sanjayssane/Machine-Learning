import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_test = np.array([0,0,0,1,0,1,1,0,1,0,0,1])
y_pred = np.array([0,0,0,0,1,1,0,0,1,1,0,0])

print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred,pos_label=0))
print(recall_score(y_test, y_pred,pos_label=1))
print(recall_score(y_test, y_pred,average='macro'))
print(recall_score(y_test, y_pred,average='weighted'))

print(precision_score(y_test, y_pred,pos_label=0))
print(precision_score(y_test, y_pred,pos_label=1))
print(precision_score(y_test, y_pred,average='macro'))
print(precision_score(y_test, y_pred,average='weighted'))

print(f1_score(y_test, y_pred,pos_label=0))
print(f1_score(y_test, y_pred,pos_label=1))
print(f1_score(y_test, y_pred,average='macro'))
print(f1_score(y_test, y_pred,average='weighted'))

print(classification_report(y_test, y_pred))




