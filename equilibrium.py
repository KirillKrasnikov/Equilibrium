"""Модуль equilibrium содержит классы для поиска конфликтных равновесий.

Содержит следующие классы:
    Equilibrium - класс для решения матричных игровых (конфликтных) задач двух лиц.
    EquilibriumOnPlane - класс для решения конфликтных хадач двух лиц на плоскости.

"""
from operator import index

# import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from decimal import Decimal
from typing import Union
# from mpl_toolkits.mplot3d import Axes3D


class Equilibrium:
    """ Класс реализует методы поиска конфликтных равновесий для матричных игр.
    Первый участник, выбирая столбец, стремится доставить максимум своей платёжной матрице j1.
    Второй участник, выбирая строку, стремится доставить максимум платёжной матрице j2.

        Поля (свойства)
        ________
            j1, j2: ndarray
                Платёжные матрицы первого и второго участников
            j_coop: ndarray
                Матрица кооперативного дохода первого и второго участников
            _a1, _a2, _a: ndarray
                Матрицы A-равновесий. Заполняются после вызова любого метода,
                который должен эти равновесия посчитать для своей работы.
            _b1, _b2, _b : ndarray
                Матрицы B-равновесий.
            _d_ol_1, __d_ol_2, _d_ol: ndarray
                Матрицы D_с_чертой-равновесий.
            _c1, _c2, _c: ndarray
                Матрицы C-равновесий.
            _cn: ndarray
                Матрица равновесий по Нэшу.
        Методы
        ______
            __init__
                Конструктор класса
            find_a_eq
                Метод поиска А-равновесных ситуаций.

    """

    def __init__(self, j1, j2, **kwargs):
        """ Конструктор класса Equilibrium, установливает значения атрибутов.

        Атрибуты
        ________
            j1, j2 : матрицы
                Платёжные матрицы первого и второго участников
        """
        self.j1 = j1
        self.j2 = j2

        self.j_coop = j1 + j2
        # Объявляем свойства для хранения матриц равновесий
        self._a1 = None
        self._a2 = None
        self._a = None
        self._b1 = None
        self._b2 = None
        self._b = None
        self._bs1 = None
        self._bs2 = None
        self._bs = None
        self._c1 = None
        self._c2 = None
        self._c = None
        self._cn = None
        self._d_ol_1 = None
        self._d_ol_2 = None
        self._d_ol = None

        self._accuracy = None
        # Устанавливаем относительную погрешность по умолчанию для функции сравнения
        # чисел с плавающей точкой iclose(a, b) = abs(a-b) < max(a,b)*rtol
        self._rtol = kwargs.get('rtol', 1e-05)
        # Устанавливаем абсолютную погрешность по умолчанию для функции сравнения
        # чисел с плавающей точкой iclose(a, b) = abs(a-b) < atol
        self._atol = kwargs.get('atol', 1e-08)

    def _almost_equal(self, a, b):
        """Метод сравнения чисел с плавающей точкой.
        """
        return np.isclose(a, b, rtol=self._rtol, atol=self._atol)

    def find_a_eq(self, visualize: bool = True):
        """Метод поиска А-равновесных ситуаций.

            Возвращаемые значения
            _____________________
                Матрицы A1, A2, A

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        szy = self.j1.shape[0]
        szx = self.j1.shape[1]

        # Выделяем память под нулевую матрицу A1 и заполняем её неопределёнными знаениями np.NaN
        # self._a1 = np.zeros(self.j1.shape)
        self._a1 = np.full(self.j1.shape, np.nan)
        # Поскольку первый участник выбирает столбец, то для того, чтобы быть A1 равновесием
        # значение в платёжной матрице должно быть больше или равно значению sup по всем столбцам
        # от inf по каждому столбцу.
        # Функция nanmin(, axis=0) возвращает вектор(массив) минимумов по каждому из столбцов, игнорируя nan.
        si = max(np.nanmin(self.j1, 0))
        max_col_idx =  np.nanmin(self.j1, 0).argmax()
        for i in range(szy):
            for j in range(szx):
                # Проверка на j1[i, j] > si для чисел с плавающей запятой с добавлением
                # проверки на равенство с учётом точности вычислений.
                   if (not np.isnan(self.j1[i, j])):
                       if (not np.isnan(self.j1[i, max_col_idx])):
                           if (self.j1[i, j] > si or self._almost_equal(self.j1[i, j], si)):
                               self._a1[i, j] = 1
                       else:
                           flag = True
                           for g in range(szx):
                               if (not np.isnan(self.j1[i, g])) and (self.j1[i, j] < np.nanmin(self.j1[:, g])):
                                   flag = False
                                   break
                           if flag:
                               self._a1[i, j] = 1


        # Выделяем память под нулевую матрицу A2
        # self._a2 = np.zeros(self.j2.shape)
        self._a2 = np.full(self.j2.shape, np.nan)
        # Поскольку второй участник выбирает строку то для того, чтобы быть A2 равновесием
        # значение в платёжной матрице должно быть больше или равно значению sup по всем строкам
        # от inf по каждой строке.
        # Функция amin(, axis=1) возвращает вектор(массив) минимумов по каждой строке.
        si = max(np.nanmin(self.j2, 1))
        for i in range(szy):
            for j in range(szx):
                # Проверка на j1[i, j] > si для чисел с плавающей запятой с добавлением
                # проверки на равенство с учётом точности вычислений.
                if (self.j2[i, j] != np.nan) and (self.j2[i, j] > si or self._almost_equal(self.j2[i, j], si)):
                    self._a2[i, j] = 1

        # Построение матрицы A-равновесий
        # Находим пересечение a1 и a2 равновесий = а-равновесию.
        # Для этого производим поэлементное умножение элементов матриц a1 и a2.
        self._a = np.multiply(self._a1, self._a2)

        return self._a1, self._a2, self._a

    def find_b_eq(self, visualize: bool = True):
        """Метод поиска B-равновесных ситуаций.

            Возвращаемые значения
            _____________________
                Матрицы B1, B2, B

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        szy = self.j1.shape[0]
        szx = self.j1.shape[1]

        # Выделяем память под нулевую матрицы B1, B2
        # self._b1 = np.zeros(self.j1.shape)
        # self._b2 = np.zeros(self.j1.shape)
        # self._b = np.zeros(self.j1.shape)
        self._b1 = np.full(self.j1.shape, np.nan)
        self._b2 = np.full(self.j2.shape, np.nan)

        # Если матрицы A-равновесий пока не построены - стоим их.
        if (self._a1 is None) or (self._a2 is None) or (self._a is None):
            self.find_a_eq(False)

        # Строим B1
        # Умножаем матрицу j2 на A1. Элементы не принадлежащие A1 имеют значение NaN и будут игнорироваться
        jb2 = np.multiply(self.j2, self._a1)
        # В вектор jb2max записываем максимумы по всем столбцам матрицы jb2, игнорируя значения NaN,
        # которые не входят в A1.
        jb2max = np.nanmax(jb2, 0)
        for j in range(szx):
            for i in range(szy):
                if self._a1[i, j] == 1:
                    # Два значения могут быть равны фактически, но при переходе к сетке
                    # значения в узлах будет отличаться,
                    # поэтому добавляем проверку на приблизительное равенство.
                    if jb2[i, j] > jb2max[j] or self._almost_equal(jb2[i, j], jb2max[j]):
                        self._b1[i, j] = 1

        # Строим B2
        # Умножаем матрицу j1 на A2. Элементы не принадлежащие A2 имеют значение NaN и будут игнорироваться
        jb1 = np.multiply(self.j1, self._a2)
        # В вектор jb1max записываем максимумы по всем строкам матрицы jb1, игнорируя значения NaN,
        # которые не входят в A2.
        jb1max = np.nanmax(jb1, 1)
        for i in range(szy):
            for j in range(szx):
                if self._a2[i, j] == 1:
                # Два значения могут быть равны фактически, но при переходе к сетке
                # значения в узлах будет отличаться,
                # поэтому добавляем проверку на приблизительное равенство.
                    if jb1[i, j] > jb1max[i] or self._almost_equal(jb1[i, j], jb1max[i]):
                        self._b2[i, j] = 1

        # Строим B, как пересечение B1 и B2
        self._b = np.multiply(self._b1, self._b2)
        return self._b1, self._b2, self._b

    def find_bs_eq(self, visualize: bool = True):
        """Метод поиска B'-равновесных ситуаций.
        Отличается от B тем, что ищается на множестве A, а не на исходном игровому множестве G.

            Возвращаемые значения
            _____________________
                Матрицы bs1, bs2, bs

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        szy = self.j1.shape[0]
        szx = self.j1.shape[1]

        # Выделяем память под нулевую матрицы B1, B2
        # self._b1 = np.zeros(self.j1.shape)
        # self._b2 = np.zeros(self.j1.shape)
        # self._b = np.zeros(self.j1.shape)
        self._bs1 = np.full(self.j1.shape, np.nan)
        self._bs2 = np.full(self.j2.shape, np.nan)

        # Если матрицы A-равновесий пока не построены - стоим их.
        if (self._a1 is None) or (self._a2 is None) or (self._a is None):
            self.find_a_eq(False)

        # Строим B'1
        # Умножаем матрицу j2 на A. Элементы не принадлежащие A имеют значение NaN и будут игнорироваться
        jb2 = np.multiply(self.j2, self._a)
        # В вектор jb2max записываем максимумы по всем столбцам матрицы jb2, игнорируя значения NaN,
        # которые не входят в A
        jb2max = np.nanmax(jb2, 0)
        for j in range(szx):
            for i in range(szy):
                if self._a[i, j] == 1:
                    # Два значения могут быть равны фактически, но при переходе к сетке
                    # значения в узлах будет отличаться,
                    # поэтому добавляем проверку на приблизительное равенство.
                    if jb2[i, j] > jb2max[j] or self._almost_equal(jb2[i, j], jb2max[j]):
                        self._bs1[i, j] = 1

        # Строим B'2
        # Умножаем матрицу j1 на A2. Элементы не принадлежащие A имеют значение NaN и будут игнорироваться
        jb1 = np.multiply(self.j1, self._a)
        # В вектор jb1max записываем максимумы по всем строкам матрицы jb1, игнорируя значения NaN,
        # которые не входят в A.
        jb1max = np.nanmax(jb1, 1)
        for i in range(szy):
            for j in range(szx):
                if self._a[i, j] == 1:
                # Два значения могут быть равны фактически, но при переходе к сетке
                # значения в узлах будет отличаться,
                # поэтому добавляем проверку на приблизительное равенство.
                    if jb1[i, j] > jb1max[i] or self._almost_equal(jb1[i, j], jb1max[i]):
                        self._bs2[i, j] = 1

        # Строим B, как пересечение B1 и B2
        self._bs = np.multiply(self._bs1, self._bs2)
        return self._bs1, self._bs2, self._bs


    def find_c_eq(self, visualize: bool = True):
        """Метод поиска C-равновесных ситуаций.
        От B равновесия оно отличаеся тем, что при нахождении Ci максимум J_j ищется не на множестве A_i, а на всём игровом множестве G.
        Для антагонистических игр двух лиц C-равновесие содержит равновесие по Нэшу СN.

            Возвращаемые значения
            _____________________
                Матрицы С1, С2, С

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        szy = self.j1.shape[0]
        szx = self.j1.shape[1]

        # Выделяем память под нулевую матрицы B1, B2
        # self._b1 = np.zeros(self.j1.shape)
        # self._b2 = np.zeros(self.j1.shape)
        # self._b = np.zeros(self.j1.shape)
        self._c1 = np.full(self.j1.shape, np.nan)
        self._c2 = np.full(self.j2.shape, np.nan)

        # Если матрицы A-равновесий пока не пострены - стоим их.
        if (self._a1 is None) or (self._a2 is None) or (self._a is None):
            self.find_a_eq(False)

        # Строим C1
        # В вектор j2max записываем максимумы по всем столбцам матрицы j2.
        j2max = np.amax(self.j2, 0)
        for j in range(szx):
            for i in range(szy):
                if self._a1[i, j] == 1:
                    # Два значения могут быть равны фактически, но при переходе к сетке
                    # значения в узлах будет отличаться,
                    # поэтому добавляем проверку на приблизительное равенство.
                    if self.j2[i, j] > j2max[j] or self._almost_equal(self.j2[i, j], j2max[j]):
                        self._c1[i, j] = 1

        # Строим C2
        # В вектор j1max записываем максимумы по всем строкам матрицы j1.
        j1max = np.amax(self.j1, 1)
        for i in range(szy):
            for j in range(szx):
                if self._a2[i, j] == 1:
                # Два значения могут быть равны фактически, но при переходе к сетке
                # значения в узлах будет отличаться,
                # поэтому добавляем проверку на приблизительное равенство.
                    if self.j1[i, j] > j1max[i] or self._almost_equal(self.j1[i, j], j1max[i]):
                        self._c2[i, j] = 1

        # Строим C, как пересечение C1 и C2
        self._c = np.multiply(self._c1, self._c2)
        return self._c1, self._c2, self._c

    def find_cn_eq(self, visualize: bool = True):
        """Метод поиска равновесия по Нэшу.  C-равновесных ситуаций.
        От С равновесия оно отличаеся тем, что при нахождении CNi, что мы проверям все точки множества G,
        а не только те, что принадлежат Ai.

            Возвращаемые значения
            _____________________
                Матрицы СN1, СN2, СN

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        szy = self.j1.shape[0]
        szx = self.j1.shape[1]

        # Выделяем память под нулевую матрицы B1, B2
        # self._b1 = np.zeros(self.j1.shape)
        # self._b2 = np.zeros(self.j1.shape)
        # self._b = np.zeros(self.j1.shape)
        cn1 = np.full(self.j1.shape, np.nan)
        cn2 = np.full(self.j2.shape, np.nan)

        # Строим CN1
        # В вектор j2max записываем максимумы по всем столбцам матрицы j2.
        j2max = np.amax(self.j2, 0)
        for j in range(szx):
            for i in range(szy):
                # Два значения могут быть равны фактически, но при переходе к сетке
                # значения в узлах будет отличаться,
                # поэтому добавляем проверку на приблизительное равенство.
                if self.j2[i, j] > j2max[j] or self._almost_equal(self.j2[i, j], j2max[j]):
                    cn1[i, j] = 1

        # Строим CN2
        # В вектор j1max записываем максимумы по всем строкам матрицы j1.
        j1max = np.amax(self.j1, 1)
        for i in range(szy):
            for j in range(szx):
                # Два значения могут быть равны фактически, но при переходе к сетке
                # значения в узлах будет отличаться,
                # поэтому добавляем проверку на приблизительное равенство.
                    if self.j1[i, j] > j1max[i] or self._almost_equal(self.j1[i, j], j1max[i]):
                        cn2[i, j] = 1

        # Строим C, как пересечение C1 и C2
        self._c = np.multiply(cn1, cn2)
        return self._c


    def find_d_ol_eq(self, visualize: bool = True):
        """Метод поиска D_с_чертой-равновесных ситуаций.
        Для нахождения D_с_чертой_1 равновесий первый участник ищет наибольшие значения своей функции полезности j1
        среди элементов множества B1.
        Аналогично ищется D_с_чертой_1.
        Множество D_с_чертой находится как пересечение множеств D_с_чертой_1 и D_с_чертой_2.

            Возвращаемые значения
            _____________________
                Матрицы d_ol_1, d_ol_1, d_ol

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        szy = self.j1.shape[0]
        szx = self.j1.shape[1]

        # Выделяем память под нулевую матрицы D_с_четрой_1, D_с_четрой_2
        self._d_ol_1 = np.full(self.j1.shape, np.nan)
        self._d_ol_2 = np.full(self.j2.shape, np.nan)

        # Если матрицы B-равновесий пока не пострены - стоим их.
        if (self._b1 is None) or (self._b2 is None) or (self._b is None):
            self.find_b_eq(False)

        # Строим D_c_чертой_1
        # Умножаем матрицу j1 на B1. Элементы не принадлежащие B1 имеют значение NaN и будут игнорироваться
        jb1 = np.multiply(self.j1, self._b1)
        # jb1max - max среди всех элементов jb1
        jb1max = np.nanmax(jb1)
        for i in range(szy):
            for j in range(szx):
                if self._b1[i, j] == 1:
                    # Два значения могут быть равны фактически, но при переходе к сетке
                    # значения в узлах будет отличаться,
                    # поэтому добавляем проверку на приблизительное равенство.
                    if self._almost_equal(jb1[i, j], jb1max):
                        self._d_ol_1[i, j] = 1
        # np.argmax

        # Строим D_c_чертой_2
        # Умножаем матрицу j2 на B2. Элементы не принадлежащие B2 имеют значение NaN и будут игнорироваться
        jb2 = np.multiply(self.j2, self._b2)
        # jb1max - max среди всех элементов jb1
        jb2max = np.nanmax(jb2)
        for i in range(szy):
            for j in range(szx):
                if self._b2[i, j] == 1:
                    # Два значения могут быть равны фактически, но при переходе к сетке
                    # значения в узлах будет отличаться,
                    # поэтому добавляем проверку на приблизительное равенство.
                    if self._almost_equal(jb2[i, j], jb2max):
                        self._d_ol_2[i, j] = 1

        # Строим D_с_чертой, как пересечение D_с_чертой_1 и D_с_чертой_2
        self._d_ol = np.multiply(self._d_ol_1, self._d_ol_2)
        return self._d_ol_1, self._d_ol_2, self._d_ol

    def get_eq_matrix(self, eq_name: str):
        """Метод возвращает матрицу равновесия, имя которой задано параметром eq_name

        Параметры
        _________
            eq_name: str
                Наименование матрицы, которую следует вернуть.
                Возможные значения: 'A1', 'A2', 'A', 'B1', и т.д.
        """
        if eq_name == 'A1':
            return self._a1
        elif eq_name == 'A2':
            return self._a2
        elif eq_name == 'A':
            return self._a
        elif eq_name == 'B1':
            return self._b1
        elif eq_name == 'B2':
            return self._b2
        elif eq_name == 'B':
            return self._b
        else:
            return None

class EquilibriumOnPlane(Equilibrium):
    """ Класс реализует методы поиска конфликтных равновесий для игр 2-x участников на плоскости

    Атрибуты
    ________
        min : float
                минимальное значение координаты по оси Ox (первого игрока)
        xmax : float
            максимальное значение координаты по оси Ox (первого игрока)
        xstep : float
            шаг дискретизации пр оси Ox
        ymin : float
            минимальное значение координаты по оси Oy (второй участник)
        ymax : float
            максимальное значение координаты по оси Oy (второй участник)
        ystep: float
            шаг дискретизации пр оси Oy

    **Защищённые поля (свойства):**
        _x : np.arange, _y np.arange
            векторы координат по осям Ox и Oy
        _xv : np.meshgrid, _yv : np.meshgrid
            сетки на игровом множестве с шагом xstep и ystep
        _xlabel:str, default: 'x'
        _ylabel:str, default: 'y'
            Наименования осей координат для рисования графиков.
        _name_1:str, default: '1'
        _name_2:str, default: '2'
            Наименования участников для обозначения равновесий на графикох.
        _rtol: float, default: 1e-05
        _atol: float, default: 2*min(xstep, ystep)
            Относительное и абсолютную погрешность для функции сравнения чисел с плавающей точкой iclose(a, b)

    Методы
    ______
    **Методы класса:**
        __init__
            Конструктор
        vizualize_utility_funcs
            Метод визуализации платёжных функций участников
    **Статические методы (могут быть вызваны непосредственно по имени класса):**
        draw_plots
            Метод выводит графики равновесий
    """

    def __init__(self, xmin, xmax, xstep, ymin, ymax, ystep, j1, j2, \
                 xlabel:str='x', ylabel:str='y', name_1:str='1', name_2:str='2', **kwargs):
        """Конструктор класса EquilibriumOnPlane, установливает значения атрибутов.

        Параметры
        _________
            xmin : float
                минимальное значение координаты по оси Ox (первого игрока)
            xmax : float
                максимальное значение координаты по оси Ox (первого игрока)
            xstep : float
                шаг дискретизации пр оси Ox
            ymin : float
                минимальное значение координаты по оси Oy (второй участник)
            ymax : float
                максимальное значение координаты по оси Oy (второй участник)
            ystep: float
                шаг дискретизации пр оси Oy
            j1 : lamda-функция, j2 : lamda-функция
                целевые функции первого и второго участника, с помощью которых генерируются
                соответствующие матрицы.
            xlabel:str, default: 'x'
            ylabel:str, default: 'y'
                Наименования осей координат для рисования графиков.
            name_1:str, default: '1'
            name_2:str, default: '2'
                Наименования участников для обозначения равновесий на графикох.
            rtol: float, default: 1e-05
            atol: float, default: 2*min(xstep, ystep)
                Относительное и абсолютную погрешность для функции сравнения чисел с плавающей точкой iclose(a, b)
        """
        self.xmin = xmin
        self.xmax = xmax
        self.xstep = xstep
        self.ymin = ymin
        self.ymax = ymax
        self.ystep = ystep
        # Создаём векторы занчений абциссы и ординаты
        # Добавляем шаг к концу диапозона, чтобы включить его в массив
        self._x = np.arange(self.xmin, self.xmax + self.xstep, self.xstep)
        self._y = np.arange(self.ymin, self.ymax  + self.ystep, self.ystep)
        # Создаём матрицы значений координат
        self._xv, self._yv = np.meshgrid(self._x, self._y, indexing='xy')

        #Задаём наименования
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._name_1 = name_1
        self._name_2 = name_2

        # Устанавливаем относительную погрешность по умолчанию для функции сравнения
        # чисел с плавающей точкой iclose(a, b) = abs(a-b) < max(a, b)*rtol
        # UPD: Значение rtol=1e-05 - Это значение по умолчанию.
        # UPD: При уменьшении относительной погрешности сильно падает точность, поэтому лучше её не менять.
        # self._rtol = kwargs.get('rtol', 1e-05)
        # Устанавливаем абсолютную погрешность по умолчанию для функции сравнения
        # чисел с плавающей точкой iclose(a, b) = abs(a-b) < atol
        # Эксперементальным путём было подобрано значение 5*min(self.xstep, self.ystep),
        # однако абсолютная погрешность может меняться в разных задачах - нужно подбирать.
        self._atol = kwargs.get('atol', 5*min(self.xstep, self.ystep))

        # Вызов конструктора базового класса и передаа в качестве параметров
        # целевых матрицы игроков, сгенерированных с помощью люмда-функций
        # Округляем значения функций до нужной величины перед занесением в матрицу
        # super().__init__(np.round(j1(self._xv, self._yv), self._accuracy), np.round(j2(self._xv, self._yv), self._accuracy))
        super().__init__(j1(self._xv, self._yv), j2(self._xv, self._yv),  atol=self._atol)

        # Определяем точность вычислений. Для этого считаем количество знаков после запятой в шаге
        # self._accuracy = min(self.xstep, self.ystep)


    def find_a_eq(self, visualize: bool = True):
        """Перегруженный метод пострения A-равновесия с параметром необходимости визуализации результатов.

        Возвращаемые значения
        _____________________
            Матрицы A1, A2, A

        Параметры
        _________
            visualize: bool, default: True
                Следует ли отображать графики равновесий.
        """
        #Вызываем метод построения равновесий для базового класса.
        super().find_a_eq()

        #Строим графики с равновесиями, если требуется
        if visualize:
            self.draw_plots(self._a1, self._a2, self._a, 12, xlabel=self._xlabel,
                            ylabel=self._ylabel, fig_title='A-равновесия',
                            title_11='$A_' + self._name_1 + '$-равновесия',
                            title_12='$A_' + self._name_2 + '$-равновесия', title_21='$A$-равновесия')

        return self._a1, self._a2, self._a

    def find_b_eq(self, visualize: bool = True):
        """Метод поиска B-равновесных ситуаций.

            Возвращаемые значения
            _____________________
                Матрицы B1, B2, B

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        # Вызываем метод построения равновесий для базового класса.
        super().find_b_eq()

        # Строим графики с равновесиями, если требуется
        if visualize:
            self.draw_plots(self._b1, self._b2, self._b, 23, xlabel=self._xlabel,
                            ylabel=self._ylabel, fig_title='B-равновесия',
                            title_11='$B_' + self._name_1 + '$-равновесия',
                            title_12='$B_' + self._name_2 + '$-равновесия', title_21='$B$-равновесия')

        return self._b1, self._b2, self._b

    def find_bs_eq(self, visualize: bool = True):
        """Метод поиска B'-равновесных ситуаций.

            Возвращаемые значения
            _____________________
                Матрицы B'1, B'2, B'

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        # Вызываем метод построения равновесий для базового класса.
        super().find_bs_eq()

        # Строим графики с равновесиями, если требуется
        if visualize:
            self.draw_plots(self._bs1, self._bs2, self._bs, 23, xlabel=self._xlabel,
                            ylabel=self._ylabel, fig_title="B'-равновесия",
                            title_11="$B'_" + self._name_1 + '$-равновесия',
                            title_12="$B'_" + self._name_2 + '$-равновесия', title_21="$B'-равновесия")

        return self._bs1, self._bs2, self._bs

    def find_c_eq(self, visualize: bool = True):
        """Метод поиска C-равновесных ситуаций.

            Возвращаемые значения
            _____________________
                Матрицы C1, C2, C

            Параметры
            _________
                visualize: bool, default: True
                    Следует ли отображать графики равновесий.
        """
        # Вызываем метод построения равновесий для базового класса.
        super().find_c_eq()

        # Строим графики с равновесиями, если требуется
        if visualize:
            self.draw_plots(self._c1, self._c2, self._c, 22, xlabel=self._xlabel,
                            ylabel=self._ylabel, fig_title='C-равновесия',
                            title_11='$C_' + self._name_1 + '$-равновесия',
                            title_12='$C_' + self._name_2 + '$-равновесия', title_21='$C$-равновесия')

        return self._c1, self._c2, self._c

    def find_d_ol_eq(self, visualize: bool = True):
        """Перегруженный метод пострения D_с_чертой-равновесия с параметром необходимости визуализации результатов.

        Возвращаемые значения
        _____________________
            Матрицы D_с_чертой_1, D_с_чертой_2, D_с_чертой

        Параметры
        _________
            visualize: bool, default: True
                Следует ли отображать графики равновесий.
        """
        #Вызываем метод построения равновесий для базового класса.
        super().find_d_ol_eq(False)

        #Строим графики с равновесиями, если требуется
        if visualize:
            self.draw_plots(self._d_ol_1, self._d_ol_2, self._d_ol, 23, xlabel=self._xlabel,
                            ylabel=self._ylabel, fig_title=r'D_с_чертой-равновесия',
                            title_11=r'$\overline{D}_' + self._name_1 + '$-равновесия',
                            title_12=r'$\overline{D}_' + self._name_2 + '$-равновесия', title_21=r'$\overline{D}$-равновесия')

        return self._d_ol_1, self._d_ol_2, self._d_ol

    def vizualize_utility_funcs(self, **kwargs):
        """Метод визуализации платёжных функций участников

        Параметры
        _____________
            **kwargs:
                fig_title : str, optional, default('Платёжная функции участников')
                    Заголовок для графика
                j1_title : str, optional, default('Платёжная функция\nпервого участника')
                    Заголовок графика платёжной функция первого участника
                j2_title : str, optional, default('Платёжная функция\nвторого участника')
                    Заголовок графика платёжной функция второго участника
                xlabel : str, optional, default('x')
                    Наименование координаты первого участника
                ylabel : str, optional, default('y')
                    Наименование координаты второго участника
        """
        title = kwargs.get('fig_title', 'Платёжная функции участников')
        fig, axes = plt.subplots(2, 2, num=title,
                                 gridspec_kw={'width_ratios': [3, 3], 'height_ratios': [3, 3]})
        # Вывод платёжной функции первогоо участника
        axes[0, 0] = plt.subplot(2, 2, 1, projection="3d")
        axes[0, 0].plot_surface(self._xv, self._yv, self.j1, vmin=self.j1.min() * 2,
                                cmap=cm.viridis, rstride=5, cstride=5,)
        plt.title(kwargs.get('j1_title', 'Платёжная функция\nпервого участника'), fontsize=11)
        axes[0, 0].set_xlabel(kwargs.get('xlabel', 'x'))
        axes[0, 0].set_ylabel(kwargs.get('ylabel', 'y'))
        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])
        axes[0, 0].margins(y=0)
        # Вывод линий уровня платёжной функции первогоо участника
        axes[1, 0] = plt.subplot(2, 2, 3)
        axes[1, 0].contour(self._xv, self._yv, self.j1)
        axes[1, 0].set_xlabel(kwargs.get('xlabel', 'x'))
        axes[1, 0].set_ylabel(kwargs.get('ylabel', 'y'))
        # Вывод платёжной функции второго участника
        axes[0, 1] = plt.subplot(2, 2, 2, projection="3d")
        axes[0, 1].plot_surface(self._xv, self._yv, self.j2, vmin=self.j2.min() * 2, cmap=cm.viridis)
        plt.title(kwargs.get('j2_title', 'Платёжная функция\nвторого участника'), fontsize=11)
        axes[0, 1].set_xlabel(kwargs.get('xlabel', 'x'))
        axes[0, 1].set_ylabel(kwargs.get('ylabel', 'y'))
        # Вывод линий уровня платёжной функции первогоо участника
        axes[1, 1] = plt.subplot(2, 2, 4)
        axes[1, 1].contour(self._xv, self._yv, self.j2)
        axes[1, 1].set_xlabel(kwargs.get('xlabel', 'x'))
        axes[1, 1].set_ylabel(kwargs.get('ylabel', 'y'))
        # Автоматический подбор отсутпов между графиками
        fig.tight_layout()
        # wspace = 0.35,
        fig.subplots_adjust(hspace=0.345)
        # Отрисовка графика
        plt.show()

    def vizualize_utility_funcs_2(self, **kwargs):
        """Метод визуализации платёжных функций участников 2

        Параметры
        _____________
            **kwargs:
                fig_title : str, optional, default('Платёжная функции участников')
                    Заголовок для графика
                j1_title : str, optional, default('Платёжная функция\nпервого участника')
                    Заголовок графика платёжной функция первого участника
                j2_title : str, optional, default('Платёжная функция\nвторого участника')
                    Заголовок графика платёжной функция второго участника
                xlabel : str, optional, default('x')
                    Наименование координаты первого участника
                ylabel : str, optional, default('y')
                    Наименование координаты второго участника
        """
        fig = plt.figure(layout="constrained")
        fig.subplots_adjust(hspace=0.35)
        # fig = plt.figure(layout="tight")
        gs0 = fig.add_gridspec(2, 1)
        # Задаём сетки для верхней и нижней частей графика
        gs00 = gs0[0].subgridspec(1, 2)
        gs01 = gs0[1].subgridspec(1, 2)
        # Рисуем линии уровня первого функционала
        ax = fig.add_subplot(gs01[0, 0])
        ax.contour(self._xv, self._yv, self.j1)
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))
        # Рисуем линии уровня второго функционала
        ax = fig.add_subplot(gs01[0, 1])
        ax.contour(self._xv, self._yv, self.j2)
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))
        # Рисуем функцию полезности первого участника
        ax = fig.add_subplot(gs00[0, 0], projection="3d")
        # Вывод платёжной функции первогоо участника
        #ax = plt.subplot(2, 2, 1, projection="3d")
        ax.plot_surface(self._xv, self._yv, self.j1, vmin=self.j1.min() * 2,
                                cmap=cm.viridis, rstride=5, cstride=5, linewidth=0, antialiased=False)
        plt.title(kwargs.get('j1_title', 'Платёжная функция\nпервого участника'), fontsize=11)
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))
        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])
        # Рисуем функцию полезности второго участника
        ax = fig.add_subplot(gs00[0, 1], projection="3d")
        # Вывод платёжной функции первогоо участника
        #ax = plt.subplot(2, 2, 1, projection="3d")
        ax.plot_surface(self._xv, self._yv, self.j2, vmin=self.j2.min() * 2,
                                cmap=cm.viridis, rstride=5, cstride=5, )
        plt.title(kwargs.get('j2_title', 'Платёжная функция\nвторого участника'), fontsize=11)
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('ylabel', 'y'))
        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])
        #Отрисовка графика
        plt.show()

    def vizualize_utility_func(self, j:np.ndarray=None, **kwargs):
        """Метод визуализации платёжной функций и её линии уровня.

        Параметры
        _____________
            j:np.ndarray, default: j1
                Функция полезности участника, которую нужно визуализировать.
            **kwargs:
                fig_title : str, optional, default('Платёжная функции участников')
                    Заголовок для графика
                j1_title : str, optional, default('Платёжная функция\nпервого участника')
                    Заголовок графика платёжной функция первого участника
                j2_title : str, optional, default('Платёжная функция\nвторого участника')
                    Заголовок графика платёжной функция второго участника
                xlabel : str, optional, default('x')
                    Наименование координаты первого участника
                ylabel : str, optional, default('y')
                    Наименование координаты второго участника
                mx, my: ndarray, optional
                    Координаты точек, которые нужно отметить на графике в виде scatterplot.
        """
        if j is None:
            j = self.j1
        title = kwargs.get('fig_title', 'Платёжные функции участников')
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 4), num=title)
        plt.suptitle(kwargs.get('j1_title', 'Платёжная функция участника {}'.format(self._name_1)), fontsize=11)
        #plt.title(kwargs.get('j1_title', 'Платёжная функция\nпервого участника'), fontsize=11)
        # Вывод платёжной функции первогоо участника
        axes[0] = plt.subplot(1, 2, 1, projection="3d")
        axes[0].plot_surface(self._xv, self._yv, j, vmin=j.min() * 2,
                                cmap=cm.viridis, rstride=5, cstride=5,)
        axes[0].set_xlabel(self._xlabel)
        axes[0].set_ylabel(self._ylabel)
        plt.xlim([self.xmin, self.xmax])
        plt.ylim([self.ymin, self.ymax])
        axes[0].margins(y=0)

        # Вывод линий уровня функции
        axes[1] = plt.subplot(1, 2, 2)
        cs = axes[1].contour(self._xv, self._yv, j)
        axes[1].clabel(cs, fontsize=9, inline=True)
        axes[1].set_xlabel(kwargs.get('xlabel', 'x'))
        axes[1].set_ylabel(kwargs.get('ylabel', 'y'))

        # -------- Отмечаем точки на графике, если нужно start -------------
        my = kwargs.get('my', None)
        mx = kwargs.get('mx', None)
        if not (mx is None or my is None):
            mz = j[my, mx]
            marker = kwargs.get('marker', '.')
            size = kwargs.get('size', 40)
            # Также определяем colormap, вес всех точек и нормализацию для их правильной раскраски
            # ListedColormap(['#e77c8d', 'w'])
            # cmap10 = cm.jet
            cmap10 = ListedColormap(['w', '#e77c8d'])
            c = np.ones(mx.shape[0])
            norm = plt.Normalize(0, 1)
            sc = axes[0].scatter(mx * self.xstep + self.xmin, my * self.ystep + self.ymin, mz, c=c,
                                   s=size, marker=marker, norm=norm, cmap=cmap10)
        # -------- Отмечаем точки на графике, если нужно end -------------


        # Автоматический подбор отсутпов между графиками
        fig.tight_layout()
        # wspace = 0.35,
        #fig.subplots_adjust(hspace=0.345)
        # Отрисовка графика
        plt.show()

    def draw_plots(self, m1, m2, m, layout:int=22, b_draw_intersect:bool=True,
                   b_draw_scatter:bool=False, **kwargs):
        """Метод выводит графики равновесий, заданных матрицами m1, m2, m

        Параметры
        _________
            m1, m2, m: ndarray
                Матрицы, которые следует вывести
            layout: int, default: 22
                Вид расположения элементов на рисунке.
                Возможные значения: 22, 12 или 23
            b_draw_intersect: bool, default: True
                Отображать ли график m графиках m1 и m2
            b_draw_scatter: bool, default: False
                Нарисовать ли все графики ввиде точечнх диагарм
        ***kwargs:*
            title_11, title_12, title_21: str, optional
                Наименования графиков
            marker: str, defalt: '.'
                Маркер для нанесения пересечения равновесий
            size: int, default: 30
                Размер маркера
        """
        if layout == 12:
            fig, (axs_m1, axs_m2, axs_cb) = plt.subplots(1, 3, figsize=(12, 3.5),
                                                         num=kwargs.get('fig_title', ''))
            axs_m = None
        elif layout == 23:
            fig, axs = plt.subplots(2, 3, figsize=(10, 5.8),
                                                        num=kwargs.get('fig_title', ''))
            axs_m1 = axs[0, 0]
            axs_m2 = axs[0, 1]
            axs_m = axs[0, 2]
            axs_cb = axs[1, 0]
            axs[1, 1].axis('off')
            axs[1, 2].axis('off')
        else:
            fig, axs = plt.subplots(2, 2, num=kwargs.get('fig_title', ''))
            axs_m1 = axs[0, 0]
            axs_m2 = axs[0, 1]
            axs_m = axs[1, 0]
            axs_cb = axs[1, 1]

        #Нужно ли рисовать график m на осях m1 и m2
        if layout == 23:
            b_draw_intersect = False


        # Определяем ненулевые элементы матрицы m, чтобы нанести их на графики m1 и m2
        (my, mx) = np.nonzero(~np.isnan(m))
        # Задаём вид маркера и его размер для точечного графика scatter
        marker = kwargs.get('marker', '.')
        size = kwargs.get('size', 30)
        # Также определяем colormap, вес всех точек и нормализацию для их правильной раскраски
        cmap10 = ListedColormap(['w', '#e77c8d'])
        c = np.ones(mx.shape[0])
        norm = plt.Normalize(0, 1)

        # Рисуем график в позиции (0, 0) #5ea5c5
        #cmap = ListedColormap(['w', '#9fd1e8'])
        cmap1 = ListedColormap(['#9fd1e8'])
        # Устанавливаем цвет для отображения NaN элементов матрицы - белый
        cmap1.set_bad('w')
        # ------------- Добавление точек и всплывающих подсказок start---------
        # Если установлен параметр b_draw_scatter, рисуем m1 в виде точечной диаграммы.
        if b_draw_scatter:
            (my1, mx1) = np.nonzero(~np.isnan(m1))
            c1 = np.ones(mx1.shape[0])
            sc1 = axs_m1.scatter(mx * self.xstep + self.xmin, my * self.ystep + self.ymin, c=c1,
                                 s=size, marker=marker, norm=norm, cmap=cmap1)
        # Иначе рисуем график ввиде pcolor
        else:
            axs_m1.pcolor(self._xv, self._yv, m1, cmap=cmap1)
        if not b_draw_scatter and b_draw_intersect:
            sc1 = axs_m1.scatter(mx * self.xstep + self.xmin, my * self.ystep + self.ymin, c=c,
                                   s=size, marker=marker, norm=norm, cmap=cmap10)
        if b_draw_scatter or b_draw_intersect:
            def hover1(event):
                vis = annot1.get_visible()
                if event.inaxes == axs_m1:
                    cont, ind = sc1.contains(event)
                    if cont:
                        EquilibriumOnPlane._update_annot(ind, sc1, annot1)
                        annot1.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot1.set_visible(False)
                            fig.canvas.draw_idle()

            annot1 = axs_m1.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                       bbox=dict(boxstyle="round", fc="w"),
                                       arrowprops=dict(arrowstyle="->"))
            annot1.set_visible(False)
            fig.canvas.mpl_connect("motion_notify_event", hover1)
        # ------------- Добавление точки и всплывающих подсказок end---------
        axs_m1.set_ylim([self.ymin - 0 * self.ystep, self.ymax + 3 * self.ystep])
        axs_m1.set_xlim([self.xmin - 0 * self.xstep, self.xmax + 3 * self.xstep])
        axs_m1.set_title(kwargs.get('title_11', ''))
        axs_m1.set_xlabel(self._xlabel)
        axs_m1.set_ylabel(self._ylabel)

        #Рисуем график в позиции (0, 1) #56ad74
        cmap2 = ListedColormap(['#8dd2a5'])
        # Устанавливаем цвет для отображения NaN элементов матрицы - белый
        cmap2.set_bad('w')
        # Если установлен параметр b_draw_scatter, рисуем m1 в виде точечной диаграммы.
        if b_draw_scatter:
            (my2, mx2) = np.nonzero(~np.isnan(m2))
            c2 = np.ones(mx2.shape[0])
            sc2 = axs_m2.scatter(mx2 * self.xstep + self.xmin, my2 * self.ystep + self.ymin, c=c2,
                                 s=size, marker=marker, norm=norm, cmap=cmap2)
        # Иначе рисуем график ввиде pcolor
        else:
            axs_m2.pcolor(self._xv, self._yv, m2, cmap=cmap2)
        # ------------- Добавление точеки и всплывающих подсказок start---------
        if not b_draw_scatter and b_draw_intersect:
            sc2 = axs_m2.scatter(mx * self.xstep + self.xmin, my * self.ystep + self.ymin, c=c,
                                    s=size, marker=marker, norm=norm, cmap=cmap10)
        if b_draw_scatter or b_draw_intersect:
            def hover2(event):
                vis = annot2.get_visible()
                if event.inaxes == axs_m2:
                    cont, ind = sc2.contains(event)
                    if cont:
                        EquilibriumOnPlane._update_annot(ind, sc2, annot2)
                        annot2.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot2.set_visible(False)
                            fig.canvas.draw_idle()

            annot2 = axs_m2.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                        bbox=dict(boxstyle="round", fc="w"),
                                        arrowprops=dict(arrowstyle="->"))
            annot2.set_visible(False)
            fig.canvas.mpl_connect("motion_notify_event", hover2)
        # ------------- Добавление точеки и всплывающих подсказок end---------
        axs_m2.set_ylim([self.ymin - 0 * self.ystep, self.ymax + 2 * self.ystep])
        axs_m2.set_xlim([self.xmin - 0 * self.xstep, self.xmax + 0 * self.xstep])
        axs_m2.set_title(kwargs.get('title_12', ''))
        axs_m2.set_xlabel(self._xlabel)
        axs_m2.set_ylabel(self._ylabel, labelpad=1.0)

        # Рисуем график в позиции (1, 0)
        if layout != 12:
            # axs_m.plot(my*self.ystep + self.ymin, mx*self.xstep + self.xmin, 'o')
            sc0 = axs_m.scatter(mx * self.xstep + self.xmin, my * self.ystep + self.ymin, c=c,
                                   s=size, marker=marker, norm=norm, cmap=cmap10)
            # ------------- Добавление всплывающие подсказки с координатой точки на графике start---------
            def hover(event):
                vis = annot0.get_visible()
                if event.inaxes == axs_m:
                    cont, ind = sc0.contains(event)
                    if cont:
                        EquilibriumOnPlane._update_annot(ind, sc0, annot0)
                        annot0.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot0.set_visible(False)
                            fig.canvas.draw_idle()
            annot0 = axs_m.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                       bbox=dict(boxstyle="round", fc="w"),
                                       arrowprops=dict(arrowstyle="->"))
            annot0.set_visible(False)
            fig.canvas.mpl_connect("motion_notify_event", hover)
            # ------------- Добавление всплывающие подсказки с координатой точки на графике end---------
            axs_m.set_ylim([self.ymin - 0 * self.ystep, self.ymax + 3 * self.ystep])
            axs_m.set_xlim([self.xmin - 0 * self.xstep, self.xmax + 3 * self.xstep])
            axs_m.set_title(kwargs.get('title_21', ''))
            axs_m.set_xlabel(self._xlabel)
            axs_m.set_ylabel(self._ylabel, labelpad=1.0)

        # Рисуем цветовую полоску с условными обозначениями #5ea5c5 #56ad74
        if axs_cb != None:
            cmap = ListedColormap(['w', '#9fd1e8', '#8dd2a5', '#e77c8d'])
            norm = BoundaryNorm([0, 0.25, 0.5, 0.75, 1], cmap.N)
            fig.colorbar(cm.ScalarMappable(norm, cmap=cmap), cax=axs_cb, location='left')
            axs_cb.set_box_aspect(20)

            # Устанавливаем наименования цветов в цветовой полоске
            yticks_pos = [0.125, 0.365, 0.625, 0.865]
            yticks_labels = ('Игровое множество', kwargs.get('title_11', ''), kwargs.get('title_12', ''),
                             kwargs.get('title_21', ''))
            axs_cb.set_yticks(yticks_pos, yticks_labels)
            axs_cb.tick_params('y', length=0.0, labelright=True, labelleft=False)

        # Устанавливаем пространство между графиками
        #fig.subplots_adjust(wspace=0.35, hspace=0.45)
        # Автоматический подбор отсутпов между графиками
        fig.tight_layout()
        plt.show()

    @staticmethod
    def _update_annot(ind, sc, annot):
        """Статический метод для создания всплывающих подсказок с координатами для точечных графиков

        Параметры
            sc: PathCollection
                Точечная диаграмма
            annot: Annotation
                Аннотация
            cmap: Colormap
            norm: Normalize
        """
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        # text = pos
        text = "({:.3f}, {:.3f})".format(pos[0], pos[1])
        annot.set_text(text)
        #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.4)



