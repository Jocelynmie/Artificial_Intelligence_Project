import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


train_X = [[1, 2], [1, 4], [7, 6], [7, 8], [9, 7]]
train_Y = [0, 0, 1, 1, 1] # categories are 0 and 1


train_X = [[1, 2], [1, 4], [7, 6], [7, 8], [9, 7]]
train_Y = [0, 0, 1, 1, 1] # categories are 0 and 1

kf = KFold(n_splits=5)

for k in [1, 2,3, 4]:
  predicted_outputs = []
  actual_outputs = []
  for (train_index, test_index) in kf.split(train_X):
    training_set_X = [train_X[i] for i in train_index]
    training_set_Y = [train_Y[i] for i in train_index]
    val_set_X = [train_X[i] for i in test_index]
    val_set_Y = [train_Y[i] for i in test_index]

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(training_set_X, training_set_Y)

    predicted_output = model.predict(val_set_X)
    predicted_outputs.extend(predicted_output)

    actual_outputs.extend(val_set_Y)

  print(f"Score for k={k} = {f1_score(actual_outputs, predicted_outputs)}")



        