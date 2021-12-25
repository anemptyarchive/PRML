### ch4.4.0 ラプラス近似

#%%

# 4.4.0項で利用するライブラリ
import numpy as np
from scipy import integrate
from scipy.stats import norm # 1変量ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%%

### 1次元の場合

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
anime_laplace.save('PRML/Fig/ch4_4_0_LaplaceApproximation.gif')

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
anime_laplace.save('PRML/Fig/ch4_4_0_LaplaceApproximation_check.gif')


#%%
