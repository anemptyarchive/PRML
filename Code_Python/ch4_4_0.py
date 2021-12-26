### ch4.4.0 ラプラス近似

#%%

### 1次元の場合

# 4.4.0項で利用するライブラリ
import numpy as np
from scipy import integrate
from scipy.stats import norm # 1変量ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%

## 真の分布の設定

# ロジスティックシグモイド関数を作成
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 真の関数を指定
def f(x):
    return np.exp(-0.5 * x**2) * sigmoid(20.0 * x + 4.0)

# 真の確率分布を指定
def p(x, f):
    # 正規化係数を計算
    C = integrate.quad(f, np.min(x), np.max(x))[0]
    return f(x) / C

# 真の関数の負の対数を指定
def E(x):
    return 0.5 * x**2 + np.log(1.0 + np.exp(-(20.0 * x + 4.0)))

# 負の対数の1階微分を指定
def nabla_E(x):
    # 中間変数を計算
    exp_x = np.exp(-(20.0 * x + 4.0))
    y = sigmoid(20.0 * x + 4.0)
    return x - 20.0 * y * exp_x

# 負の対数の2階微分を指定
def nabla2_E(x):
    # 中間変数を計算
    exp_x = np.exp(-(20.0 * x + 4.0))
    y = sigmoid(20.0 * x + 4.0)
    return 1.0 + 400.0 * y**2 * exp_x


# 真の関数の1階微分を指定
def nabla_f(x):
    # シグモイド関数の計算
    y = sigmoid(20.0 * x + 4.0)
    return np.exp(-0.5 * x**2) * y * (20.0 * (1.0 - y) - x)

# 作図用のxの点を作成
x_vals = np.arange(-2.0, 4.0, 0.01)

#%%

# 真の分布のグラフを作成
plt.figure(figsize=(8, 8))
plt.plot(x_vals, f(x_vals), color='purple', label='$f(x)$') # 真の関数
plt.plot(x_vals, p(x_vals, f), color='blue', label='$p(x)$') # 真の分布
plt.xlabel('x')
plt.ylabel('f(x), p(x)')
plt.title('$f(x) = \exp(- 0.5 x^2) \sigma(20 x + 4)$', fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%

# 負の対数のグラフを作成
plt.figure(figsize=(8, 8))
plt.plot(x_vals, E(x_vals), color='purple', label='$E(x)$') # 負の対数
plt.plot(x_vals, nabla_E(x_vals), color='purple', linestyle='--', label='$\\nabla E(x)$') # 1階微分
plt.plot(x_vals, nabla2_E(x_vals), color='purple', linestyle=':', label='$\\nabla^2 E(x)$') # 2階微分
plt.plot(x_vals, p(x_vals, f), color='blue', label='$p(x)$') # 真の分布
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('$E(x) = - \ln f(x)$', fontsize=20)
plt.legend()
plt.grid()
#plt.ylim(0.0, 1.0)
plt.show()

#%%

## 最頻値の推定

# 試行回数を指定
max_iter = 100

# 最急降下法の学習係数を指定
eta = 0.05

# xの初期値を指定
x = 3.0

# 最頻値を探索
trace_x = np.zeros(max_iter + 1)
trace_x[0] = x
for i in range(max_iter):
    # 負の対数の1階微分を計算
    dy = nabla_E(x)
    
    # 負の対数の2階微分を計算
    d2y = nabla2_E(x)
    
    # xを更新
    x -= eta * dy # 最急降下法
    #x -= dy / d2y # ニュートン-ラフソン法
    
    # 値を記録
    trace_x[i+1] = x.copy()

# 最頻値を記録(近似分布の平均を設定)
x0 = x.copy()
print(x0)

#%%

# 更新値の推移を作図
plt.figure(figsize=(8, 8))
plt.plot(x_vals, E(x_vals), color='purple', label='$E(x)$') # 負の対数
plt.plot(trace_x, E(trace_x), c='chocolate', marker='o', mfc='orange', mec='orange', label='$x^{(t)}$') # xの推移
plt.scatter(x0, E(x0), color='red', marker='x', s=200, label='$x^{(' + str(max_iter) +')}$') # 最頻値の推定値
#plt.plot(x_vals, f(x_vals), color='blue', label='$E(x)$') # 真の関数
#plt.scatter(x0, f(x0), color='red', marker='x', s=200, label='$x^{(' + str(max_iter) +')}$') # xの推移:(真の関数)
#plt.plot(trace_x, f(trace_x), c='chocolate', marker='o', mfc='orange', mec='orange', label='$x^{(t)}$') # 最頻値の推定値:(真の関数)
plt.xlabel('x')
plt.ylabel('E(x)')
plt.suptitle('Gradient Descent, Newton-Raphson Method', fontsize=20)
plt.title('iter:' + str(max_iter) + ', x=' + str(np.round(x, 3)), loc='left')
plt.legend()
plt.grid()
plt.show()

#%%

## 近似分布の計算

# 近似分布の平均を指定(手計算)
#x0 = 0.06

# 近似分布の精度を計算
A = nabla2_E(x0)
print(A)

# 近似分布の確率密度を計算
laplace_dens = norm.pdf(x=x_vals, loc=x0, scale=np.sqrt(1.0 / A))

#%%

# 近似分布のグラフを作成
plt.figure(figsize=(8, 8))
plt.plot(x_vals, p(x_vals, f), color='blue', label='p(x)') # 真の分布
plt.plot(x_vals, laplace_dens, color='darkturquoise', label='q(x)') # 近似分布
plt.vlines(x=x0, ymin=0.0, ymax=np.max([p(x_vals, f), laplace_dens]), colors='gray', linestyle=':', label='$\mu = x_0$') # 最頻値の推定値
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle('Laplace Aporroximation', fontsize=20)
plt.title('$\mu=' + str(np.round(x0, 2)) + 
          ', \sigma=' + str(np.round(np.sqrt(1.0 / A), 2)) + '$', loc='left')
plt.legend()
plt.grid()
plt.show()

#%%

# 負の対数のグラフを作成
plt.figure(figsize=(8, 8))
plt.plot(x_vals, -np.log(p(x_vals, f)), color='blue', label='$- \ln p(x)$') # 真の分布の対数
plt.plot(x_vals, -np.log(laplace_dens), color='darkturquoise', label='$- \ln q(x)$') # 近似分布の対数
plt.xlabel('x')
plt.ylabel('log density')
plt.title('Negative logarithm', fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%

# 図を初期化
fig = plt.figure(figsize=(8, 8))

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のxの値を取得(近似分布の平均を設定)
    x0 = trace_x[i]
    
    # 近似分布の精度を計算
    A = nabla2_E(x0)
    
    # 近似分布の確率密度を計算
    laplace_dens = norm.pdf(x=x_vals, loc=x0, scale=np.sqrt(1.0 / A))
    
    # 近似分布のグラフを作成
    plt.plot(x_vals, p(x_vals, f), color='blue', label='p(x)') # 真の分布
    plt.plot(x_vals, laplace_dens, color='darkturquoise', label='q(x)') # 近似分布
    plt.vlines(x=x0, ymin=0.0, ymax=np.max([p(x_vals, f), laplace_dens]), colors='gray', linestyle=':', label='$\mu = x_0$') # 最頻値の推定値
    plt.xlabel('x')
    plt.ylabel('density')
    plt.suptitle('Laplace Aporroximation', fontsize=20)
    plt.title('$iter:' + str(i) + 
              ', \mu=' + str(np.round(x0, 3)) + 
              ', \sigma=' + str(np.round(np.sqrt(1.0 / A), 3)) + '$', loc='left')
    plt.legend()
    plt.grid()

# gif画像を作成
anime_laplace = FuncAnimation(fig, update, frames=max_iter, interval=100)

# gif画像を保存
anime_laplace.save('PRML/Fig/ch4_4_0_LaplaceApproximation1D.gif')

#%%

# 図を初期化
fig = plt.figure(figsize=(8, 8))

# x_valsからフレームに利用する間隔を指定
n = 3

# y軸の表示範囲を指定
y_min = -0.5
y_max = 1.5

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のxの値を取得(近似分布の平均を設定)
    x0 = x_vals[i*n]
    
    # 近似分布の精度を計算
    A = nabla2_E(x0)
    
    # 近似分布の確率密度を計算
    laplace_dens = norm.pdf(x=x_vals, loc=x0, scale=np.sqrt(1.0 / A))
    
    # 近似分布のグラフを作成
    plt.plot(x_vals, p(x_vals, f), color='blue', label='p(x)') # 真の分布
    plt.plot(x_vals, laplace_dens, color='darkturquoise', label='q(x)') # 近似分布
    plt.vlines(x=x0, ymin=y_min, ymax=y_max, colors='gray', linestyle=':', label='$\mu = x_0$') # 推定した最頻値
    plt.scatter(x0, 0.0, color='gray') # 推定した最頻値
    plt.plot(x_vals, nabla_E(x_vals), color='purple', linestyle='--', label='$\\nabla E(x)$') # 1階微分
    plt.scatter(x0, nabla_E(x0), color='purple') # 1階微分
    plt.plot(x_vals, np.sqrt(1.0 / nabla2_E(x_vals)), color='purple', linestyle=':', label='$\\sigma = \\sqrt{\\frac{1}{A}}$') # 標準偏差
    plt.scatter(x0, np.sqrt(1.0 / nabla2_E(x0)), color='purple') # 標準偏差
    plt.xlabel('x')
    plt.ylabel('density')
    plt.suptitle('Laplace Aporroximation', fontsize=20)
    plt.title('$\mu=' + str(np.round(x0, 3)) + 
              ', \sigma=' + str(np.round(np.sqrt(1.0 / A), 3)) + '$', loc='left')
    plt.legend()
    plt.grid()
    plt.ylim(y_min, y_max)

# gif画像を作成
anime_laplace = FuncAnimation(fig, update, frames=int(len(x_vals) / n), interval=100)

# gif画像を保存
anime_laplace.save('PRML/Fig/ch4_4_0_LaplaceApproximation1D_check.gif')


#%%

### 2次元の場合

# 4.4.0項で利用するライブラリ
import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal # 多変量ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%

## 真の分布の設定

# 真の関数を指定
def f(X0, X1):
    return 0.25 * np.sin(X0) -  0.2 * np.cos(X1) + 0.5

# 真の確率分布を指定
def p(X0, X1, f):
    # 正規化係数を計算
    C = integrate.dblquad(f, np.min(X0), np.max(X0), lambda x: np.min(X1), lambda x: np.max(X1))[0]
    return f(X0, X1) / C

# 負の対数を作成
def E(X0, X1):
    return - np.log(f(X0, X1))

# 負の対数の勾配(階微分)を作成
def nabla_E(x0, x1):
    # 中間変数を計算
    denom = - (5.0 * np.sin(x0) - 4.0 * np.sin(x1) + 10.0)
    vec = np.array([5.0 * np.cos(x0), 4.0 * np.sin(x1)])
    return vec / denom

# 負の対数のヘッセ行列(2階微分)を作成
def nabla2_E(x0, x1):
    # 中間変数を計算
    denom = 5.0 * np.sin(x0) - 4.0 * np.cos(x1) + 10.0
    mat = np.zeros((2, 2))
    mat[0, 0] = 5.0 * np.sin(x0) * denom + 25.0 * np.cos(x0)**2
    mat[0, 1] = 20.0 * np.cos(x0) * np.sin(x1)
    mat[1, 0] = 20.0 * np.cos(x0) * np.sin(x1)
    mat[1, 1] = - 4.0 * np.cos(x1) * denom + 16.0 * np.sin(x1)**2
    return mat / denom**2


# 作図用のxの値を作成
x0_vals = np.arange(-3.0, 4.0, 0.1)
x1_vals = np.arange(-6.0, 1.0, 0.1)

# 作図用のxの点を作成
X0, X1 = np.meshgrid(x0_vals, x1_vals)
x_point = np.stack([X0.flatten(), X1.flatten()], axis=1)
x_dims = X0.shape
print(x_dims)

#%%

# 真の関数のグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X0, X1, f(X0, X1), cmap='jet') # 真の関数:(3D)
ax.contour(X0, X1, f(X0, X1), cmap='jet', offset=np.min(f(X0, X1))) # 真の関数:(等高線)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$f(x)$')
ax.set_title('$f(x) = \\frac{1}{4} \sin x_0 - \\frac{1}{5} \cos x_1 + 0.5$', fontsize=20)
plt.show()

#%%

# 真の分布のグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X0, X1, p(X0, X1, f), cmap='jet') # 真の分布:(3D)
ax.contour(X0, X1, p(X0, X1, f), cmap='jet', offset=0.0) # 真の分布:(等高線)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('density')
ax.set_title('$p(x) = \\frac{f(x)}{\int f(x) dx}$', fontsize=20)
plt.show()

#%%

# 負の対数のグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X0, X1, E(X0, X1), cmap='jet') # 負の対数:(3D)
ax.contour(X0, X1, E(X0, X1), cmap='jet', offset=0.0) # 負の対数:(等高線)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$E(x)$')
ax.set_title('$E(x) = - \ln f(x)$', fontsize=20)
plt.show()

#%%

## 最頻値の推定

# 試行回数を指定
max_iter = 100

# 最急降下法の学習係数を指定
eta = 0.5

# xの初期値を指定
x_d = np.array([-1.0, -0.5])

# 最頻値を探索
trace_x = np.zeros((max_iter + 1, 2))
trace_x[0] = x_d.copy()
for i in range(max_iter):
    # 負の対数の勾配(1階微分)を計算
    grad = nabla_E(x_d[0], x_d[1])
    
    # 負の対数のヘッセ行列(2階微分)を計算
    H = nabla2_E(x_d[0], x_d[1])
    
    # xを更新
    #x_d -= eta * grad # 最急降下法
    x_d -= np.dot(np.linalg.inv(H), grad.reshape(2, 1)).flatten() # ニュートン-ラフソン法
    
    # 値を記録
    trace_x[i+1] = x_d.copy()

# 最頻値を記録(近似分布の平均を設定)
x0_d = x_d.copy()
print(x0_d)

#%%

# 更新値の推移を作図
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X0, X1, E(X0, X1), cmap='jet', alpha=0.5) # 負の対数:(3D)
ax.contour(X0, X1, E(X0, X1), cmap='jet', linestyles=':', offset=0.0) # 負の対数:(等高線)
ax.plot(trace_x[:, 0], trace_x[:, 1], E(trace_x[:, 0], trace_x[:, 1]), c='chocolate', marker='o', mfc='orange', mec='orange', label='$x^{(t)}$') # xの推移
ax.scatter(x_d[0], x_d[1], E(x_d[0], x_d[1]), color='red', marker='x', s=200, label='$x^{(' + str(max_iter) +')}$') # 推定した最頻値
ax.plot(trace_x[:, 0], trace_x[:, 1], 0, c='chocolate', linestyle='--') # xの推移:(平面)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('$E(x)$')
ax.set_title('iter:' + str(max_iter) + 
             ', x=(' + ', '.join(str(x) for x in np.round(x_d, 3)) + ')', loc='left')
fig.suptitle('Gradient Descent, Newton-Raphson Method', fontsize=20)
ax.legend()
plt.show()

#%%

## 近似分布の計算

# 近似分布の平均を指定(手計算)
#x0_d = np.array([0.0, 0.0])

# 近似分布の精度行列を計算
A_dd = nabla2_E(x0_d[0], x0_d[1])
print(A_dd)
print(np.linalg.inv(A_dd))

# 近似分布の確率密度を計算
laplace_dens = multivariate_normal.pdf(x=x_point, mean=x0_d, cov=np.linalg.inv(A_dd))

#%%

# 近似分布のグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_wireframe(X0, X1, p(X0, X1, f), alpha=0.3) # 真の関数:(3D)
ax.contour(X0, X1, p(X0, X1, f), cmap='jet', linestyles='--', offset=0) # 真の関数:(等高線)
ax.plot_surface(X0, X1,laplace_dens.reshape(x_dims), cmap='jet', alpha=0.9) # 近似分布:(3D)
ax.contour(X0, X1, laplace_dens.reshape(x_dims), cmap='jet', offset=0) # 近似分布:(等高線)
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.set_zlabel('density')
ax.set_title('$iter:' + str(max_iter) + 
             ', \mu=(' + ', '.join([str(x) for x in np.round(x0_d , 3)]) + ')' + 
             ', \Sigma=[' + ', '.join([str(sgm) for sgm in np.round(A_dd, 3)]) + ')$', loc='left')
fig.suptitle('Laplace Approximation', fontsize=20)
plt.show()

#%%

# 分散が負の値になると計算できないので問題ない試行番号をチェック
for i in reversed(range(max_iter)):
    x_d = trace_x[i]
    A_dd = nabla2_E(x_d[0], x_d[1])
    laplace_dens = multivariate_normal.pdf(x=x_point, mean=x0_d, cov=np.linalg.inv(A_dd))
    print(i)

#%%

# 図を初期化
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Laplace Approximation', fontsize=20)

# フレームを開始する試行番号を指定
n = 4

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のxの値を取得
    x0_d = trace_x[i+n]
    
    # 近似分布の精度行列を計算
    A_dd = nabla2_E(x0_d[0], x0_d[1])

    # 近似分布の確率密度を計算
    laplace_dens = multivariate_normal.pdf(x=x_point, mean=x0_d, cov=np.linalg.inv(A_dd))

    # 条件付きガウス分布の3Dグラフを作成
    ax.plot_wireframe(X0, X1, p(X0, X1, f), alpha=0.3) # 真の関数:(3D)
    ax.contour(X0, X1, p(X0, X1, f), cmap='jet', linestyles='--', offset=0) # 真の関数:(等高線)
    ax.plot_surface(X0, X1,laplace_dens.reshape(x_dims), cmap='jet', alpha=0.9) # 近似分布:(3D)
    ax.contour(X0, X1, laplace_dens.reshape(x_dims), cmap='jet', offset=0) # 近似分布:(等高線)
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('density')
    ax.set_title('$iter:' + str(i+n) + 
                 ', \mu=(' + ', '.join([str(x) for x in np.round(x0_d , 3)]) + ')' + 
                 ', \Sigma=[' + ', '.join([str(sgm) for sgm in np.round(np.linalg.inv(A_dd), 3)]) + ')$', loc='left')

# gif画像を作成
anime_laplace = FuncAnimation(fig, update, frames=(max_iter+1-n), interval=100)

# gif画像を保存
anime_laplace.save('PRML/Fig/ch4_4_0_laplasApproximation2D.gif')


#%%

print('end')

