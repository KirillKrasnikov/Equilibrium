import numpy as np
import equilibrium

def check_k_eq():
    """Поиск индивидуально-кооперативного понятия равносеия
        """

    j1 = np.array([[5, np.nan, 2, 9], [11, 12, 10, np.nan], [np.nan, 1, 10, 4], [8, 6, np.nan, 3]])
    j2 = np.array([[8, np.nan, 10, 6], [1, 4, 12, np.nan], [np.nan, 7, 2, 9], [3, 11, np.nan, 5]])


    matrix_game = equilibrium.Equilibrium(j1, j2)
    print("j1 = \n", matrix_game.j1)
    print("j2 = \n", matrix_game.j2)
    print("j = \n", matrix_game.j_coop)
    (a1, a2, a) = matrix_game.find_a_eq()
    print("a1 = \n", a1)
    print("a2 = \n", a2)
    print("a = \n", a)

if __name__ == '__main__':
    #   Основная функция
    check_k_eq()


