import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from dtree import load_data

X_train, y_train = load_data('train')
X_val, y_val = load_data('val')
X_test, y_test = load_data('test')

# clf = RandomForestClassifier(criterion='entropy', n_jobs=-1, n_estimators=100, max_features=0.2, oob_score=True)
# clf.fit(X_train, y_train)

parameters = {
    'n_estimators' : list(range(50, 450, 100)),
    'max_features' : [0.1, 0.3, 0.5, 0.7, 0.9],
    'min_samples_split' : [2, 4, 6, 8, 10],
    'oob_score' : [True]
}

# parameters = {
#     'n_estimators' : [50],
#     'max_features' : [0.1],
#     'min_samples_split' : [2, 4],
#     'oob_score' : [True]
# }

random_forest = RandomForestClassifier()
clf = GridSearchCV(random_forest, parameters, n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)

f = open("scikit-dtree.txt","w")
print(clf.best_params_, file=f)
print(clf.best_score_, file=f)
print('',file=f)
print(clf.cv_results_, file=f)
f.close()

print("All done")