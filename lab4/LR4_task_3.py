import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from utilities import visualize_classifier
import pandas as pd
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)



parameter_grid = [{'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]}, 
                  {'n_estimators': [4], 'max_depth': [25, 50, 100, 250]}]


metrics = ['precision_weighted', 'recall_weighted']
for metric in metrics:
    print("\n#### Searching optimal parameters for", metric)
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

print("\nGrid scores for the parameter grid:")

df = pd.DataFrame(classifier.cv_results_)
df_columns_to_print = [column for column in df.columns if 'param' in column or 'score' in column]
print(df[df_columns_to_print])

print("\nBest parameters:", classifier.best_params_)

y_pred = classifier.predict(X_test)
print("\nPerformance report:\n")
print(classification_report(y_test, y_pred))
