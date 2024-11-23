import numpy as np
from common import assert_ndarray_equal, load_test_data, draw_cotour
from os.path import abspath, dirname, join
from solution import *


def test_classifier_1():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=1.0, kernel='linear')
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.74, "Test 1 Failed"

      

def test_classifier_2():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=1.0, kernel='poly', kernel_parameter=2.0)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.70, "Test 2 Failed"
      

def test_classifier_3():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=1.0, kernel='poly', kernel_parameter=3.0)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.74, "Test 3 Failed"
      

def test_classifier_4():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=1.0, kernel='rbf', kernel_parameter=2.0)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.75, "Test 4 Failed"


def test_classifier_5():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=0.01, kernel='linear')
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.70, "Test 5 Failed"

      

def test_classifier_6():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=0.01, kernel='poly', kernel_parameter=2.0)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.75, "Test 6 Failed"
      

def test_classifier_7():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=0.01, kernel='poly', kernel_parameter=3.0)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  assert accuracy > 0.75, "Test 7 Failed"
      

def test_classifier_8():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'test'))

  trainer = NonLinearDualSVM(C=0.01, kernel='rbf', kernel_parameter=2.0)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)
  draw_cotour(trainer, X_test, y_test, './images/NonLinearSVM.jpg', num=51)

  assert accuracy > 0.70, "Test 8 Failed"