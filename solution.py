import numpy as np
from scipy import optimize


class BinaryEstimatorSVM:
    """
    Класс для построения модели бинарной классификации методом опорных 
    векторов путем решения прямой задачи оптимизации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).

    fit_intercept : bool, по умолчанию True
        Включать ли свободный член (сдвиг) в модель.

    drop_last : bool, по умолчанию True
        Удалять ли последний неполный батч из обучения.

    coef_ : numpy.ndarray или None
        Коэффициенты (веса) модели размером (n_features, 1), которые обучаются на данных. Инициализируются как None до вызова метода `fit`.

    intercept_ : numpy.ndarray или None
        Свободный член (сдвиг) модели размером (1). Инициализируется как None до вызова метода `fit`.

    n_classes_ : int
        Количество классов.

    """

    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):
        """
        Инициализация объекта класса LinearPrimalSVM с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.

        drop_last : bool, по умолчанию True
            Удалять ли последний неполный батч из обучения.
        """

        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.coef_ = None
        self.intercept_ = None
        self.n_classes_ = None


    def predict(self, X):
        """
        Предсказывает расстояние до разделяющей классы гиперплоскости для входных данных на основе обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных расстояний, ориентированных по нормали к разделяющей гиперплоскости.
        """

        return np.dot(X, self.coef_) + (self.intercept_ if self.fit_intercept else 0)


    def loss(self, X, y_true):
        """
        Вычисляет функцию потерь для бинарной классификации на основе HingeLoss
        с учетом L2 регуляризации

        Параметры:
        ----------
        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов.

        Возвращает:
        ----------
        float
            Значение функции потерь.
        """

        predict = self.predict(X)
        return np.mean(np.maximum(0, 1 - y_true * predict.T)) + 0.5 * self.C * np.sum(self.coef_ ** 2)
      
    def loss_grad(self, X, y_true):
        """
        Вычисляет градиент функции потерь по отношению к весам модели.

        В случае использования регуляризации, градиент включает соответствующие компоненты для
        штрафа за большие значения весов.

        Параметры:
        ----------

        X : numpy.ndarray
            Входной массив признаков размером (n_samples, n_features), где n_samples — количество образцов,
            а n_features — количество признаков.

        y_true : numpy.ndarray
            Вектор истинных меток классов.

        Возвращает:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели.

        grad_intercept : numpy.ndarray
            Градиент функции потерь по отношению к свободному члену.
        """

        filterr = (1 - y_true * self.predict(X).T) > 0
        return -np.dot(X.T, (y_true * filterr).T) + self.C * self.coef_, -np.sum(y_true * filterr) if self.fit_intercept else None


    def step(self, grad, grad_intercept):
        """
        Выполняет один шаг обновления весов модели с использованием вычисленного градиента.

        Параметры:
        ----------
        grad : numpy.ndarray
            Градиент функции потерь по отношению к весам модели (размером как coef_).
        
        grad_intercept : numpy.ndarray или None
            Градиент функции потерь по отношению к свободному члену (размером как intercept_).
            Если fit_intercept=False, этот параметр будет равен None.

        Возвращает:
        ----------
        None
        """
        self.coef_ -= self.lr * grad
        if self.fit_intercept:
            self.intercept_ -= self.lr * grad_intercept
        return

    def fit(self, X, y):
        """
        Обучает модель SVM с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : LinearPrimalSVM
            Обученная модель.
        """

        self.coef_ = np.zeros(X.shape[1])

        self.intercept_ = 0.0 if self.fit_intercept else None

        for _ in range(self.n_epochs):
            els = list(np.random.permutation(X.shape[0]))
            X_sh = X[els]
            y_sh = y[els]

            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_sh[i:min(i + self.batch_size, X.shape[0])]
                y_batch = y_sh[i:min(i + self.batch_size, X.shape[0])]

                grad, grad_intercept = self.loss_grad(X_batch, y_batch)

                self.step(grad, grad_intercept)

            if X.shape[0] % self.batch_size != 0 and not self.drop_last:
                X_batch = X_sh[X.shape[0] // self.batch_size * self.batch_size:]
                y_batch = y_sh[X.shape[0] // self.batch_size * self.batch_size:]
                grad, grad_intercept = self.loss_grad(X_batch, y_batch)

                self.step(grad, grad_intercept)
 
        return self


def one_vs_rest(y, n_classes=None):
    """
    Преобразует целевые метки в матрицу, где метки целевого класса
    принимают значение 1, а остальные метки — значение -1.

    Параметры:
    ----------
    y : numpy.ndarray или list
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    -----------
    numpy.ndarray
        Двумерная матрица размером (n_samples, n_classes), где для каждого образца целевой
        класс представлен значением 1, а все остальные классы имеют значение -1.

    """

    y = np.array(y)
    n_classes = np.max(y) + 1 if not n_classes else n_classes
    ans = -1 * np.ones((y.shape[0], n_classes))

    for i in range(y.shape[0]):
        ans[i, y[i]] = 1

    return ans


class LinearPrimalSVM:
    """
    Класс для построения модели многоклассовой классификации методом опорных 
    векторов путем решения прямой задачи оптимизации.

    Параметры:
    ----------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).
    
    Атрибуты:
    ---------
    lr : float, default=0.01
        Скорость обучения (learning rate) для обновления коэффициентов модели.

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    n_epochs : int, default=100
        Количество эпох для обучения модели.

    batch_size : int, default=16
        Размер батча для mini-batch градиентного спуска (mini-batch GD).

    fit_intercept : bool, по умолчанию True
        Включать ли свободный член (сдвиг) в модель.

    drop_last : bool, по умолчанию True
        Удалять ли последний неполный батч из обучения.

    self.n_classes_ : int
        Количество классов, определяемое на основе уникальных меток в обучающем наборе данных.
        Этот параметр устанавливается после вызова метода `fit` и используется для определения 
        размерности выходного пространства модели. Он равен максимальному значению метки в данных плюс один.

    list_of_models : list
        Список, содержащий бинарные модели.
    """

    def __init__(self, lr=0.01, C=1.0, n_epochs=100, batch_size=16, fit_intercept=True, drop_last=True):
        """
        Инициализация объекта класса LinearPrimalSVM с заданными гиперпараметрами.

        Параметры:
        ----------
        lr : float, default=0.01
            Скорость обучения (learning rate) для обновления коэффициентов модели.

        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        n_epochs : int, default=100
            Количество эпох для обучения модели.

        batch_size : int, default=16
            Размер батча для mini-batch градиентного спуска (mini-batch GD).

        fit_intercept : bool, по умолчанию True
            Включать ли свободный член (сдвиг) в модель.

        drop_last : bool, по умолчанию True
            Удалять ли последний неполный батч из обучения.
        """

        self.lr = lr
        self.C = C
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        self.drop_last = drop_last
        self.n_classes_ = None
        self.list_of_models = []


    def predict(self, X):
        """
        Предсказывает метки классов для входных данных на основе обученной модели.

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        ----------
        numpy.ndarray
            Вектор предсказанных меток классов (значения от 0 до n_classes-1).
        """
        
        arr = []
        for m in self.list_of_models:
            arr.append(m.predict(X))
        return np.argmax(np.array(arr), axis=0).reshape((X.shape[0], ))

    def tmp(self, X, onevrest):
        for i in range(onevrest.shape[1]):
            m = BinaryEstimatorSVM(lr=self.lr, C=self.C, n_epochs=self.n_epochs, batch_size=self.batch_size, fit_intercept=self.fit_intercept, drop_last=self.drop_last)

            m.fit(X, onevrest[:, i])
            self.list_of_models.append(m)

    def fit(self, X, y):
        """
        Обучает модель SVM с использованием mini-batch градиентного спуска (mini-batch GD).

        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        ----------
        self : LinearPrimalSVM
            Обученная модель.
        """

        onevrest = one_vs_rest(y)
        self.tmp(X, onevrest)
        
        return self
    
    
def kernel_linear(x1, x2):
    """
    Линейное ядро для SVM.

    Вычисляет скалярное произведение двух векторов, что соответствует линейной 
    границе разделения в пространстве признаков.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.

    Возвращает:
    ----------
    float
        Скалярное произведение векторов x1 и x2.
    """

    return np.dot(x1, x2)


def kernel_poly(x1, x2, d=2):
    """
    Полиномиальное ядро для SVM.

    Вычисляет полиномиальное скалярное произведение двух векторов, 
    что позволяет моделировать нелинейные границы разделения.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.
    
    d : int, default=2
        Степень полинома.

    Возвращает:
    ----------
    float
        Полиномиальное скалярное произведение векторов x1 и x2.
    """

    return (kernel_linear(x1, x2) + 1) ** d


def kernel_rbf(x1, x2, l=1.0):
    """
    Радиально-базисное (гауссовское) ядро для SVM.

    Вычисляет расстояние между двумя векторами с использованием радиально-базисной функции (RBF),
    которая позволяет моделировать сложные нелинейные зависимости.

    Параметры:
    ----------
    x1 : np.array, shape (n_features,)
        Первый вектор признаков.
    
    x2 : np.array, shape (n_features,)
        Второй вектор признаков.
    
    l : float, default=1.0
        Параметр ширины гауссовской функции (коэффициент сглаживания).

    Возвращает:
    ----------
    float
        Значение RBF-ядра между векторами x1 и x2.
    """

    norm = -np.linalg.norm(x1 - x2) ** 2
    return np.exp(0.5 * norm / (l ** 2))

def lagrange(gramm_matrix, alpha):
    """
    Двойственная функция Лагранжа для SVM.

    Вычисляет двойственную функцию для оптимизации SVM с использованием
    заранее рассчитанной матрицы Грамма.

    Параметры:
    ----------
    gramm_matrix : np.array, shape (n_samples, n_samples)
        Матрица Грамма (значения ядер между всеми парами обучающих объектов).
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда), используемые для оптимизации.

    Возвращает:
    ----------
    float
        Значение двойственной функции Лагранжа.
    """

    summ = alpha.sum()
    multi = np.dot(alpha, gramm_matrix)
    return summ - 0.5 * np.dot(alpha, multi)

def lagrange_derive(gramm_matrix, alpha):
    """
    Производная двойственной функции Лагранжа по alpha.

    Вычисляет градиент (производную) двойственной функции Лагранжа,
    что необходимо для решения задачи оптимизации.

    Параметры:
    ----------
    gramm_matrix : np.array, shape (n_samples, n_samples)
        Матрица Грама (значения ядер между всеми парами обучающих объектов).
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда), используемые для оптимизации.

    Возвращает:
    ----------
    np.array, shape (n_samples,)
        Градиент двойственной функции по alpha.
    """

    multi = np.dot(alpha, gramm_matrix)
    return np.ones(alpha.shape) - multi

def one_vs_one(X, y, n_classes=None):
    """
    Преобразует целевые метки в матрицу, где метки первого класса
    принимают значение 1, а метки второго — значение -1.

    Параметры:
    ----------
    y : numpy.ndarray
        Вектор или список меток классов, которые необходимо закодировать.
        Значения меток должны быть целыми числами от 0 до n_classes-1.

    n_classes : int или None, по умолчанию None
        Количество классов (размерность выходного пространства).
        Если None, то количество классов определяется автоматически как максимум значения в y плюс один.

    Возвращает:
    -----------
    list of tuples
        (X_cut, y_cut (Бинарный таргет 1 или -1), соответствующий '1' класс, соответствующий '-1' класс)
        
    """

    n_classes = np.max(y) + 1 if not n_classes else n_classes
    ans = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            filterr = (y == i) | (y == j)
            X1 = X[filterr]
            y1 = np.where(y[filterr] == i, 1, -1)
            ans.append((X1, y1, i, j))
    return ans


class SoftMarginSVM:
    """
    Реализация SVM с мягким зазором (Soft Margin SVM) с возможностью использовать произвольные ядра.
    
    Атрибуты:
    ----------
    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.
    
    alpha : np.array, shape (n_samples,)
        Двойственные переменные (лямбда) для решения задачи оптимизации.

    supportVectors : np.array, shape (n_support_vectors, n_features)
        Опорные вектора — обучающие объекты, которые оказывают влияние на разделяющую гиперплоскость.

    supportLabels : np.array, shape (n_support_vectors,)
        Метки классов для опорных векторов.

    supportalpha : np.array, shape (n_support_vectors,)
        Значения альфа (лямбда) для опорных векторов.

    kernel : function
        Ядро для вычисления скалярных произведений в пространстве признаков.

    classes_names : list or array-like, shape (2,)
        Имена классов. Используются для преобразования предсказанных значений {-1, 1} в имена классов.
    
    b: float 
        Смещение.

    Методы:
    -------
    fit(X, y):
        Обучает модель.

    predict(X):
        Предсказывает метки классов для входных данных.
    """
    
    def __init__(self, C, kernel_func, classes_names=None):
        """
        Инициализирует модель Soft Margin SVM.
        
        Параметры:
        ----------
        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.
        
        kernel_func : function
            Функция ядра, определяющая метод вычисления скалярных произведений в новом пространстве признаков.
        
        classes_names : list
            Список имен классов. Ожидается, что в обучающих данных метки классов {-1, 1}.
        """
        self.C = C                                 
        self.alpha = None
        self.supportVectors = None
        self.supportLabels = None
        self.supportalpha = None
        self.kernel = kernel_func
        self.classes_names = classes_names
        self.b = None

    
    def fit(self, X, y):
        """
        Обучает модель с использованием оптимизации двойственной задачи для SVM.
        
        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Обучающие данные (матрица признаков).
        
        y : np.array, shape (n_samples,)
            Вектор меток классов, должен содержать значения {-1, 1}.
        
        Возвращает:
        ----------
        self : SoftMarginSVM
            Обученная модель.
        """

        N = len(y)
        
        kernalized_X = np.apply_along_axis(lambda x1 : np.apply_along_axis(lambda x2: self.kernel(x1, x2), 1, X), 1, X)  
        GramXy = np.dot(kernalized_X, np.matmul(y.reshape((y.shape[0], 1)), y.reshape((1, y.shape[0]))))

        first_constraint = {'type': 'eq', 'fun': lambda a: np.dot(a, y), 'jac': lambda a: y}

        A = np.vstack((-np.eye(N), np.eye(N)))

        b = np.hstack((np.zeros(N), self.C * np.ones(N)))

        second_constraint = {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A}
        constraints = (first_constraint, second_constraint)

        opt_res = optimize.minimize(fun=lambda a: -lagrange(GramXy, a),
                                    x0=np.ones(N),
                                    method='SLSQP',
                                    jac=lambda a: -lagrange_derive(GramXy, a),
                                    constraints=constraints
                                    )

        self.alpha  = opt_res.x

        epsilon = 1e-6

        valid_indices = (self.alpha > epsilon) & (self.alpha <= self.C + epsilon)
        self.supportVectors = X[valid_indices]
        self.supportLabels = y[valid_indices]
        self.supportalpha = self.alpha

        if len(self.supportVectors) > 0:
            b_values = []
            
            for x_k, y_k in zip(self.supportVectors, self.supportLabels):

                tmp = 0
                for i in range(self.supportVectors.shape[0]):
                    tmp += self.supportLabels[i] * self.supportalpha[i] * self.kernel(x_k, self.supportVectors[i])
                b_values.append(y_k - tmp)
            self.b = np.mean(b_values)
        else:
            self.b = 0
        
        return self

    def ker(self, X, i):
        ker = 0
        for j in range(self.supportVectors.shape[0]):
            ker += self.kernel(X[i], self.supportVectors[j]) * self.supportalpha[j] * self.supportLabels[j]

        ker += self.b
        return ker

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных.
        
        Параметры:
        ----------
        X : np.array, shape (n_samples, n_features)
            Массив объектов для предсказания.
        
        Возвращает:
        ----------
        np.array, shape (n_samples,)
            Вектор предсказанных меток классов, где метки соответствуют значениям из `classes_names`.
        """
        
        ans = []
        n_samples = X.shape[0]

        for i in range(n_samples):
            ans.append(self.ker(X, i))
        
        return np.where(np.array(ans) > 0, float(self.classes_names[0]), float(self.classes_names[1])).reshape((np.array(ans).shape[0], 1))

class NonLinearDualSVM:
    """
    NonLinearDualSVM реализует SVM one-vs-one с использованием двойственной задачи. 
    Поддерживает использование различных ядерных функций для задач классификации.

    Атрибуты:
    ---------
    estimators : list или None
        Список бинарный SVM моделей one-vs-one

    C : float, default=1.0
        Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

    kernel : str, default='rbf'
        Ядерная функция, используемая в модели (возможные значения: 'poly', 'rbf', 'linear').

    """

    def __init__(self, C=1.0, kernel='rbf', kernel_parameter=1.0):
        """
        Инициализирует модель NonLinearDualSVM с указанной ядерной функцией.
        
        Параметры:
        ----------
        C : float, default=1.0
            Коэффициент, контролирующий баланс между минимизацией ошибок классификации и максимизацией зазора.

        kernel : str, default='rbf'
            Ядерная функция, используемая в модели (возможные значения: 'poly', 'rbf', 'linear').

        kernel_parameter : float, default=1.0
            Гиперпарметр ядра
      
        """
        self.C = C

        if kernel == 'poly':
          self.kernel = lambda x, y: kernel_poly(x, y, d=kernel_parameter)
        elif kernel == 'rbf':
          self.kernel = lambda x, y: kernel_rbf(x, y, l=kernel_parameter)
        else:
          self.kernel = kernel_linear

        self.kernel.__name__='kernel'

        self.estimators = []

    def final_decision(self, vote_counts):
        return np.apply_along_axis(lambda preds: float(np.bincount(preds).argmax()), axis=1, arr=vote_counts)

    def predict(self, X):
        """
        Предсказывает метки классов для входных данных X.
        
        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Входные данные для предсказания меток классов.

        Возвращает:
        -------
        numpy.ndarray, shape (n_samples,)
            Предсказанные метки классов для каждого образца.
        """

        num_samples = X.shape[0]
        num_models = len(self.estimators)
        vote_counts = np.zeros((num_samples, num_models), dtype=int)

        for idx, model in enumerate(self.estimators):
            model_predictions = model.predict(X)
            vote_counts[:, idx] = model_predictions.flatten()

        return self.final_decision(vote_counts)


    def fit(self, X, y):
        """
        Обучает модель SVM на тренировочных данных (X, y) с использованием двойственной задачи.
        
        Параметры:
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Тренировочные данные.

        y : numpy.ndarray, shape (n_samples,)
            Целевые метки классов.

        Возвращает:
        -------
        self : NonLinearDualSVM
            Обученная модель
        """
        
        self.estimators += [SoftMarginSVM(C=self.C, kernel_func=self.kernel, classes_names=list((cl1, cl2))).fit(X1, y1) for X1, y1, cl1, cl2 in one_vs_one(X, y)]
        return self
