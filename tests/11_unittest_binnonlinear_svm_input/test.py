import numpy as np
from common import assert_ndarray_equal, load_test_data, draw_cotour
from os.path import abspath, dirname, join
from solution import *


def test_classifier_1():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=1.0, kernel_func=lambda x, y: kernel_linear(x, y), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)

  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.85, "Test 1 Failed"

      

def test_classifier_2():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=1.0, kernel_func=lambda x, y: kernel_poly(x, y, d=2.0), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.75, "Test 2 Failed"
      

def test_classifier_3():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=1.0, kernel_func=lambda x, y: kernel_poly(x, y, d=3.0), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.80, "Test 3 Failed"
      

def test_classifier_4():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=1.0, kernel_func=lambda x, y: kernel_rbf(x, y, l=2.0), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.85, "Test 4 Failed"


def test_classifier_5():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=0.01, kernel_func=lambda x, y: kernel_linear(x, y), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.85, "Test 5 Failed"

      

def test_classifier_6():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=0.01, kernel_func=lambda x, y: kernel_poly(x, y, d=2.0), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.85, "Test 6 Failed"
      

def test_classifier_7():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=0.01, kernel_func=lambda x, y: kernel_poly(x, y, d=3.0), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.90, "Test 7 Failed"
      

def test_classifier_8():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  X_train = X_train[y_train <= 1]
  X_test = X_test[y_test <= 1]
  y_train = y_train[y_train <= 1]
  y_test = y_test[y_test <= 1]
  y_train[y_train != 1] = -1
  y_test[y_test != 1] = -1

  trainer = SoftMarginSVM(C=0.01, kernel_func=lambda x, y: kernel_rbf(x, y, l=2.0), classes_names=(1, 0))
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred.reshape(-1) == (y_test+1)/2)

  assert accuracy > 0.85, "Test 8 Failed"