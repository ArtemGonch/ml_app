import numpy as np
from common import assert_ndarray_equal, load_test_data, draw_cotour
from os.path import abspath, dirname, join
from solution import *


def test_classifier_1():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_1'))

  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  model = BinaryEstimatorSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  model = model.fit(X_train, y_train)
  y_pred =  model.predict(X_test)

  accuracy = np.mean(np.sign(y_pred)==y_test)

  assert accuracy > 0.55, "Test 1 Failed"
      

def test_classifier_2():
  data_path = __file__.replace("test.py", "")
  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_2'))

  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  model = BinaryEstimatorSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  model = model.fit(X_train, y_train)
  y_pred =  model.predict(X_test)
  accuracy = np.mean(np.sign(y_pred)==y_test)

  assert accuracy > 0.70, "Test 2 Failed"
      

def test_classifier_3():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_3'))
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  model = BinaryEstimatorSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  model = model.fit(X_train, y_train)

  y_pred =  model.predict(X_test)
  accuracy = np.mean(np.sign(y_pred)==y_test)

  assert accuracy > 0.65, "Test 3 Failed"      
      