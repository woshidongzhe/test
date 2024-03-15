import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体，可根据需要更换为其他中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

# 定义目标函数
def obj_fun1(x):
    return 11*np.sin(x) + 7*np.cos(5*x)

# 绘制函数图像
x = np.arange(-3, 3, 0.1)
y = obj_fun1(x)
plt.figure()
plt.plot(x, y, 'b-')

# 参数初始化
narvs = 1  # 变量个数
T0 = 100   # 初始温度
T = T0     # 迭代中温度会发生改变，第一次迭代时温度就是T0
maxgen = 200  # 最大迭代次数
Lk = 100  # 每个温度下的迭代次数
alfa = 0.95  # 温度衰减系数
x_lb = -3  # x的下界
x_ub = 3  # x的上界

# 随机生成初始解
x0 = np.random.uniform(x_lb, x_ub, narvs)
y0 = obj_fun1(x0)
h = plt.scatter(x0, y0, c='r', marker='*')

# 模拟退火过程
max_y = y0
MAXY = np.zeros(maxgen)
for iter in range(maxgen):
    for i in range(Lk):
        y = np.random.randn(narvs)  # 生成1行narvs列的N(0,1)随机数
        z = y / np.sqrt(np.sum(y**2))  # 根据新解的产生规则计算z
        x_new = x0 + z*T  # 根据新解的产生规则计算x_new的值
        x_new = np.clip(x_new, x_lb, x_ub)  # 如果这个新解的位置超出了定义域，就对其进行调整

        x1 = x_new
        y1 = obj_fun1(x1)
        if y1 > y0:
            x0 = x1
            y0 = y1
        else:
            p = np.exp(-(y0 - y1)/T)
            if np.random.rand() < p:
                x0 = x1
                y0 = y1

        if y0 > max_y:
            max_y = y0
            best_x = x0

    MAXY[iter] = max_y
    T = alfa*T

    h.set_offsets(np.c_[x0, obj_fun1(x0)])
    plt.pause(0.01)

# 输出最佳位置和最优值
print('最佳的位置是：', best_x)
print('此时最优值是：', max_y)

h.remove()
plt.scatter(best_x, max_y, c='r', marker='*')
plt.title('模拟退火找到的最大值为 {}'.format(max_y))

# 画出每次迭代后找到的最大y的图形
plt.figure()
plt.plot(range(1, maxgen+1), MAXY, 'b-')
plt.xlabel('迭代次数')
plt.ylabel('y的值')
plt.show()
