import numpy as np
from common import assert_ndarray_equal, load_test_data, draw_cotour
from os.path import abspath, dirname, join
from solution import *


def test_classifier_1():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_1'))

  trainer = LinearPrimalSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  draw_cotour(trainer, X_test, y_test, './images/LinearSVM1.jpg', num=501)

  assert accuracy > 0.95, "Test 1 Failed"
      

def test_classifier_2():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_2'))

  trainer = LinearPrimalSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)
  draw_cotour(trainer, X_test, y_test, './images/LinearSVM2.jpg', num=501)
  assert accuracy > 0.74, "Test 2 Failed"
      

def test_classifier_3():
  data_path = __file__.replace("test.py", "")

  X_train, X_test, y_train, y_test = load_test_data(join(data_path, 'linear_test_3'))

  trainer = LinearPrimalSVM(lr=0.01, C=0.01, n_epochs=1000, batch_size=128, fit_intercept=True, drop_last=True)
  trainer = trainer.fit(X_train, y_train)
  y_pred =  trainer.predict(X_test)
  accuracy = np.mean(y_pred==y_test)

  draw_cotour(trainer, X_test, y_test, './images/LinearSVM3.jpg', num=501)

  assert accuracy > 0.85, "Test 3 Failed"      
      