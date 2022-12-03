import numpy as np
from math import *
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)

# Длина отрезка по x
lx = 100
# Длина отрезка по y
ly = 100
# Число разбиений по x
Nx = 100
# Число разбиений по y
Ny = 100
# Шаг по x
dx = lx/Nx
# Шаг по y
dy = ly/Ny
# k
k = 0.5
# tau
tau = 1
# Число итераций по времени
Tpred = 30


# Истиное решение
def h(x, y, t):
    return (cos(2*pi*(x+t)/lx) * cos(2*pi*y/ly) )
    #return x + y + t


# Значения на нулевом слое
def f0(x, y):
    return h(x, y, 0)


# Граничные условия
def f1(y, t):
    return h(0, y, t)
def f2(y, t):
    return h(lx, y, t)
def f3(x, t):
    return h(x, 0, t)
def f4(x, t):
    return h(x, ly, t)


# Функция H(x,y) - поверхность
def H(x,y):
    return 4


def a(ht,i, j):
    return 1/2 * (H(i*lx/Nx, j*ly/Ny) + ht[i][j] + H((i-1)*lx/Nx, j*ly/Ny) + ht[i-1][j])
def b(ht,i, j):
    return 1/2 * (H(i*lx/Nx, j*ly/Ny)+ ht[i][j] + H(i*lx/Nx, (j-1)*ly/Ny) + ht[i][j-1])


# Первый слой
def FirstLayer():
    x = np.zeros((Nx + 1, Ny + 1))
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            x[i][j] = f0(i*lx/Nx, j*ly/Ny)
    return x


# Разностная схема 1
def Matrix1(ht,j, t):
    A = np.zeros((Nx + 1, Nx + 1))
    for i in range (1, Nx):
        A[i][i - 1] = - k * a(ht,i,j) / dx**2
        A[i][i] = 2/tau + k / (dx**2)*a(ht,i+1,j) +k/(dx**2)*a(ht,i,j)
        A[i][i + 1] = - k/(dx**2)*a(ht,i+1,j)
    A[0][0] = 1
    A[Nx][Nx] = 1
    return A


# Разностная схема 2
def Matrix2(ht,i, t):
    A = np.zeros((Ny + 1, Ny + 1))
    for j in range (1, Ny):
        A[j][j - 1] = - k * b(ht,i,j) / dy**2
        A[j][j] = 2/tau + k / (dy**2)*b(ht,i,j+1) + k/(dy**2)*b(ht,i,j)
        A[j][j + 1] = - k/(dy**2)*b(ht,i,j+1)
    A[0][0] = 1
    A[Ny][Ny] = 1
    return A


# Правая часть для схемы 1
def Vector1(ht, j, t):
    a = np.zeros(Nx + 1)
    for i in range (1, Nx):
        a[i] = 2* ht[i][j] / tau + k/dy * \
               (b(ht,i,j+1) * (ht[i][j+1]-ht[i][j]) / dy - b(ht,i,j) * (ht[i][j]-ht[i][j-1]) / dy)
    a[0] = f1(j*ly/Ny, t*tau)
    a[Nx] = f2(j*ly/Ny, t*tau)
    return a


# Правая часть для схемы 2
def Vector2(ht, i, t):
    b = np.zeros(Ny + 1)
    for j in range (1, Ny):
        b[j] = 2* ht[i][j] / tau + k/dx * \
               (a(ht,i+1,j) * (ht[i+1][j]-ht[i][j]) / dx - a(ht,i,j) * (ht[i][j]-ht[i-1][j]) / dx)
    b[0] = f3(i*lx/Nx, t*tau)
    b[Ny] = f4(i*lx/Nx, t*tau)
    return b


# Метод прогонки
def Progonka(A, f):
    N = len(f)
    gamma = np.zeros(N)
    betta = np.zeros(N)
    alpha = np.zeros(N)
    x = np.zeros(N)
    gamma[0] = A[0][0]
    betta[0] = f[0] / gamma[0]
    alpha[0] = - A[0][1] / gamma[0]
    for i in range (1, N - 1):
        gamma[i] = A[i][i] + A[i][i - 1] * alpha[i - 1]
        betta[i] = (f[i] - A[i][i - 1] * betta[i - 1]) / gamma[i]
        alpha[i] = - A[i][i + 1] / gamma[i]
    gamma[N - 1] = A[N - 1][N - 1] + A[N - 1][N - 2] * alpha[N - 2]
    betta[N - 1] = (f[N - 1] - A[N - 1][N - 2] * betta[N - 2]) / gamma[N - 1]
    x[N - 1] = betta[N - 1]
    for i in range(N - 2, - 1, - 1):
        x[i] = alpha[i] * x[i + 1] + betta[i]
    return x


# Запись значений на границах
def SetBorders(h, t):
    for j in range(0, Ny+1):
        h[0][j] = f1(j*ly/Ny, t*tau)
        h[Nx][j] = f2(j*ly/Ny, t*tau)
    for i in range(0, Nx+1):
        h[i][0] = f3(i*lx/Nx,t*tau)
        h[i][Ny] = f4(i*lx/Nx,t*tau)


# Численное решение
def Resh():
    h = FirstLayer()
    h1 = np.zeros((Nx+1, Ny+1))
    for t in range(0, Tpred):
        SetBorders(h1, t + 1 / 2)
        for j in range(1, Ny):
            A = Matrix1(h,j,t+1/2)
            bb = Vector1(h,j,t+1/2)
            hh = Progonka(A,bb)
            h1[j] = hh
        h1 = np.transpose(h1)
        h = np.copy(h1)
        SetBorders(h1, t + 1)
        for i in range(1, Nx):
            A = Matrix2(h,i,t+1)
            bb = Vector2(h,i,t+1)
            hh = Progonka(A,bb)
            h1[i] = hh
        h = np.copy(h1)

    return h


# Точное решение
def TochResh():
    x = np.zeros((Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(Ny+1):
            x[i][j] = h(i*lx/Nx, j*ly/Ny, Tpred*tau)
    return x


# Вывод численного решения
def Show(res):
    xi = np.linspace(0, lx, Nx + 1)
    yi = np.linspace(0, ly, Ny + 1)
    fig = plt.figure(figsize=(8, 5))
    plot1 = fig.add_subplot(1, 1, 1, projection = '3d', title = "h(x,y,"+str(Tpred) + ")")
    x, y = np.meshgrid(xi, yi)
    surf = plot1.plot_surface(x, y, res, rstride = 1, cstride = 1, cmap = 'ocean')
    plt.show()


# Вывод численного решения
def ShowDiff(res):
    xi = np.linspace(0, lx, Nx + 1)
    yi = np.linspace(0, ly, Ny + 1)
    fig = plt.figure(figsize=(8, 5))
    plot1 = fig.add_subplot(1, 2, 1, projection = '3d', title = "Численное решение")
    plot2 = fig.add_subplot(1, 2, 2, projection='3d', title="Точное решение")
    x, y = np.meshgrid(xi, yi)
    surf = plot1.plot_surface(x, y, res, rstride = 1, cstride = 1, cmap = 'ocean')
    toch = TochResh()
    surf = plot2.plot_surface(x, y, toch, rstride=1, cstride=1, cmap='ocean')
    fig.suptitle("f(x,y,"+str(Tpred)+"); ||h_toch-h_chisl|| = " + str(np.linalg.norm(toch-res)), fontsize=15)
    plt.show()

res = Resh()
Show(res)
#ShowDiff(res)