import matplotlib.pyplot as plt # библиотека для отображения графиков
import numpy as np # библиотека для работы с массивами и матрицами
import math # библиотека для математических функций
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter


eps = input('Эпсилон ')
a_1 = input('a_1 ')
p = input('Время * 10 ')
k = input('Точность - 1/h ')
k = int(k)
eps = float(eps) # эпсилон
h = 1/k # шаг алгоритма
n = k*10 # число итерации
X = np.zeros((n, 2)) # матрица n на 2 заполненная нулями для решения системы (2)
g = 1 # сигма
a_0 = g * math.sqrt(1 - g**2 / 4) # константа a_0
a_1 = int(a_1) # константа a_1
a = a_0 * (1 + eps**2 * a_1)
w = math.sqrt(1-(g**2)/2) # константа для начальных значений
f_2 = 0 # функция из уравнения
f_3 = -1 # функция из уравнения
p = int(p) # Число полных повторений алгоритва на одном наборе дынных для снятия ограничений по памяти

def Initial_Values(): # задаем начальные значения для нашего уравнения (1)
    c = 0
    for t in np.arange(-1, 0, h):
        X[c][0] = math.cos((w/eps)*t)*eps
        X[c][1] = math.sin((w/eps)*t)*eps
        c+=1
    return X

def Initial_Saved_Values(): # задаем начальные значения для нашего уравнения (1) из файла с сохраненными точками
    with open('C:/saved_data[0.029-5-1000].npy', 'rb') as f:
        a = np.load(f)
    for i in range(0,k):
        X[i][0] = a[i][0]
        X[i][1] = a[i][1]
    return X

def I_Values(): # сохраняем k послених значений для повторения алгоритма с данными начальными значениями
    for i in range(0,k):
        X[i][0] = X[n - k + i][0]
        X[i][1] = X[n - k + i][1]
    return X

def Dinamics(): # основная функция        
    for _ in range(p): # повторяем весь алгоритм при p > 1
        for i in range(k,n): # шаг алгоритма, используем метод Рунге - Кутты 2 порядка
            df = Df(X[i-1],X[i-k]) # X[i-100] запаздывание
            X[i] = X[i-1] + h*df
            X[i] = X[i-1] + h*df*(1/2) + h*(1/2)*Df(X[i],X[i-k+1])
        if (p > 1):
            I_Values()
    return X

def Df(X,X_t): # находим решение системы (5) для каждого шага метода Рунге - Кутты
    x = X[0]
    y = X[1]
    x_t = X_t[0]
    dx = y
    dy = (-g*eps*y - x + a*x_t + f_2*x_t**2 + f_3*x_t**3)/eps**2            
    z = np.array([dx, dy])
    return z

Initial_Values() # вызов функций либо заново считаем либо загружаем сохраненные данные
# Initial_Saved_Values()
m = Dinamics()

plt.rcParams.update({'font.size': 25})
fig = plt.figure()
ax = fig.gca()

plt.xlabel("t")  # рисуем график X(t)
plt.ylabel("x(t)")

plt.plot(np.linspace(int(n*h*p - n*h),int(n*h*p),n-k),m[k:,0],label = "eps = %s, a_1 = %d, time = %d" % (eps,a_1,n*h*p))
plt.legend()
fig.set_figheight(8)
fig.set_figwidth(15)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.2f"))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)   
plt.savefig('C:/saved_figure_1[%s-%d-%d-%d].png'%(eps,a_1,p*10,f_2)) # сохраняем график
plt.show()

plt.xlabel("X") # рисуем график Y(X)
plt.ylabel("Y")
plt.plot(m[:,0],m[:,1],label = "%s-%d-%d" % (eps,a_1,p*10))
plt.grid()
plt.legend()   
plt.savefig('C:/saved_figure_2[%s-%d-%d].png'%(eps,a_1,p*10))
plt.show()

save = True # сохраняем посчитанные точки 
if save == True:
    with open('C:/saved_data[1].npy', 'wb') as f:
        np.save(f, m[n-k:])