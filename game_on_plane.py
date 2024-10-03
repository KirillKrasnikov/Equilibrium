import numpy as np
import equilibrium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

def pursuit_game_static():
    """Решение дифференциальной задачи преследования на одном интервале планирования (статическая
    задача).
    """
    # Ввод начальных значений параметров
    xmin = -1.57
    xmax = 1.57
    xstep = 0.01
    ymin = -1
    ymax = 1
    ystep = xstep

    # x_0 = 5.0
    # y_0 = 4.0
    x_0 = 0.01873270872009676
    y_0 = 1.1476658701559543
    w = 1.5
    t0 = 0
    t1 = 0.06669921080659218

    # Функция, максимум которой ищет участник E
    hp = lambda x, y: x_0 * w * np.cos(x) - y_0 * y + y_0 * w * np.sin(x) - \
                      (w ** 2 + y ** 2 - 2 * w * y * np.sin(x))*(t1 - t0)
    he = lambda x, y: -hp(x, y)
    # Функция, максимум которой ищет участник P

    # xlabel = '$\phi$', ylabel = '$\psi$', fig_title = 'B-равновесия',
    # title_11 = '$A_P$-равновесия', title_12 = '$A_E$-равновесия',
    # title_21 = '$A$-равновесия'

    # rtol = 1e-05,
    pursuit_game = equilibrium.EquilibriumOnPlane(xmin, xmax, xstep, ymin, ymax, ystep, hp, he,
                                                  xlabel='$\phi$', ylabel = '$\psi$', name_1='P', name_2='E',
                                                  atol=0.00001)
    pursuit_game.vizualize_utility_func(pursuit_game.j2,
                                        j1_title='Гамильтониан $H$, который минимизирует $P$\nи максимизирует $E$')

    # pursuit_game.find_a_eq()
    # pursuit_game.find_b_eq()
    # pursuit_game.find_bs_eq()
    # pursuit_game.find_d_ol_eq()
    (c1, c2, c) = pursuit_game.find_c_eq(True)
    # cn = pursuit_game.find_cn_eq()
    # pursuit_game.draw_plots(c, cn, cn, 12, marker='*', b_draw_scatter=True,
    #                         fig_title='C-равновесие и равновесие по Нэшу',
    #                         title_11='C-равновесие', title_12='Равновесие по Нэшу')
    # # (my, mx) = np.nonzero(~np.isnan(c))
    # pursuit_game.vizualize_utility_func(pursuit_game.j2,
    #                                     j1_title='Гамильтониан $H$, который минимизирует $P$\nи максимизирует $E$',
    #                                     mx=mx, my=my)

def pursuit_game_dynamic():
    """Решение задачи преследований в динамике с анимацией"""

    xmin = -1.57
    xmax = 1.57
    xstep = 0.01
    ymin = -1
    ymax = 1
    ystep = xstep

    x_0 = 5.0
    y_0 = 4.0
    w = 1.5

    # Устанавливаем временные интервалы для одной итерации и
    # количество шагов (установки точек на графике) за один временной промежуток
    t0 = 0
    t1 = math.sqrt(x_0**2 + y_0**2) / w
    frame_per_t = 32
    t = np.linspace(t0, t1, frame_per_t)

    # Функция, максимум которой ищет участник E
    hp = lambda x, y: x_0 * w * np.cos(x) - y_0 * y + y_0 * w * np.sin(x) - \
                      (w ** 2 + y ** 2 - 2 * w * y * np.sin(x))*(t1 - t0)
    he = lambda x, y: -hp(x, y)


    phi = np.arange(xmin, xmax + xstep, xstep)
    psi = np.arange(ymin, ymax + ystep, ystep)

    frames = 0
    # Устанавливаем критерий завершения задачи преследования - расстояние между участниками
    while math.sqrt(x_0**2 + y_0**2) >= 1:
        # Устанавливаем значения начальных параметров, соответсвующих положению объектов в конце
        # предыдущего периода
        pursuit_game = equilibrium.EquilibriumOnPlane(xmin, xmax, xstep, ymin, ymax, ystep, hp, he,
                                                      xlabel='$\phi$', ylabel='$\psi$', name_1='P', name_2='E',
                                                      atol=xstep / 1000)
        (c1, c2, c) = pursuit_game.find_c_eq(False)
        (my, mx) = np.nonzero(~np.isnan(c))

        # Удаляем старый объект
        del pursuit_game

        # Если на очередном шаге не удалось найти рановесия по Нэшу или координата Px стала <= 0
        # уменьшаем шаг по времени в два раза и рассчитываем равновесия заново для новой задачи
        # с уменьшенным интервалом планирования
        if len(my) == 0:
            print('Пересчёт траектории:  x_0={}, y_0={}, t1={}'.format(x_0, y_0, t1))
            # Уменьшаем в два раза интервал планирования t и шаг внутри интервала frame_per_t
            if t1 > 0.01:
                t1 /= 2
                frame_per_t //= 2
                t = np.linspace(t0, t1, frame_per_t)
                continue
            # Если шаг по времени уже слишком мал - перестаём считать
            else:
                break
        # Если координата Px стала отрицательной - тоже уменбшеам интервал планирования
        elif x_0 - w * np.cos(phi[mx[0]])*t1 <= 0:
            print('Пересчёт траектории: x_0={}, y_0={}, t1={}, x_0[-1]={}'.format(x_0, y_0, t1,
                                                             x_0 - w * np.cos(phi[mx[0]])*t1))
            # Уменьшаем в два раза интервал планирования t и шаг внутри интервала frame_per_t
            if t1 > 0.1:
                t1 /= 2
                frame_per_t //= 2
                t = np.linspace(t0, t1, frame_per_t)
                continue
            # Если шаг по времени уже слишком мал - перестаём считать
            else:
                break

        # Траектория движения убегающего E
        # На первой итерации создаём массив точек, описывающей положения E первом интервале времени
        if frames == 0:
            e_pos_y = y_0 + psi[my[0]] * t
            e_pos_x = 0.1 + 0 * t
        # На последующих итерациях добавляем к существующему массиву положения на новой итерации
        else:
            e_pos_y = [*e_pos_y, *(e_pos_y[-1] + psi[my[0]] * t)]
            e_pos_x = [*e_pos_x, *(0.1 + 0 * t)]

        # Траектория движения преследователя P
        # На первой итерации создаём массив точек, описывающей положения E первом интервале времени
        if frames == 0:
            p_pos_y = w * np.sin(phi[mx[0]]) * t
            p_pos_x = x_0 - w * np.cos(phi[mx[0]]) * t
        # На последующих итерациях добавляем к существующему массиву положения на новой итерации
        else:
            p_pos_y = [*p_pos_y, *(p_pos_y[-1] + w * np.sin(phi[mx[0]]) * t)]
            p_pos_x = [*p_pos_x, *(p_pos_x[-1] - w * np.cos(phi[mx[0]]) * t)]

        # Устанавливаем новые значения относительных координат для новой итерации
        y_0 = e_pos_y[-1] - p_pos_y[-1]
        x_0 = p_pos_x[-1]

        print('Состояние в конце итерации:  x_0={}, y_0={}, t1={}, |P-E|={}'.format(x_0,
                                                                                    y_0, t1,
                                                                                    math.sqrt(x_0**2 + y_0**2)))

        frames += frame_per_t

    # Строим анимированный график движения участников
    fig, ax = plt.subplots(1, 1)

    line_e = ax.plot(e_pos_x[0], e_pos_y[0], label='Убегающий E', linestyle='dashed')[0]
    line_p = ax.plot(p_pos_x[0], p_pos_y[0], label='Преследователь P')[0]
    ax.legend()

    ax.set(xlim=[0, 13], ylim=[0, 13], xlabel='x', ylabel='y')
    # plt.xticks(np.arange(0, 13, 1))
    plt.xticks([0, 0.3, 0.6, 0.8, 1, 1.3,  1.6, 2, 2.3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
               ['0', '', '', '', '1', '',  '', '2', '', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'])
    plt.yticks(np.arange(0, 13, 1))

    def update(frame):
        # Обновление данных графиков на каждой итерации:
        line_e.set_xdata(e_pos_x[:frame])
        line_e.set_ydata(e_pos_y[:frame])

        line_p.set_xdata(p_pos_x[:frame])
        line_p.set_ydata(p_pos_y[:frame])
        return (line_e, line_p)

    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=60, repeat=False)

    writervideo = animation.FFMpegWriter(fps=10)
    ani.save('pursuit_game_5.mp4', writer=writervideo)
    plt.show()

if __name__ == '__main__':
    #   Основная функция

    pursuit_game_static()
    #   pursuit_game_dynamic()