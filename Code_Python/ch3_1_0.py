### ch3.1.0 基底関数

# 3.1.0項で利用するライブラリ
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%%

### 1次元の場合

## 基底関数の作成

# 多項式基底関数を作成
def phi_poly(x, j):
    return x**j

# ガウス基底関数を作成
def phi_gauss(x, mu, s):
    return np.exp(-(x - mu)**2 / (2.0 * s**2))

# シグモイド基底関数を作成
def phi_sigmoid(x, mu, s):
    a = (x - mu) / s
    return 1.0 / (1.0 + np.exp(-a))

#%%

# 作図用のxの点を作成
x_vals = np.linspace(-1.0, 1.0, 201)

#%%

# パラメータを指定
j = 3
mu = 0.0
s = 0.1

# 基底関数を作図
plt.figure(figsize=(12, 9))
#plt.plot(x_vals, phi_poly(x_vals, j), label='j=' + str(j)) # 多項式基底関数
#plt.plot(x_vals, phi_gauss(x_vals, mu, s), label='$\mu=' + str(mu) + ', s=' + str(s) + '$') # ガウス基底関数
plt.plot(x_vals, phi_sigmoid(x_vals, mu, s), label='$\mu=' + str(mu) + ', s=' + str(s) + '$') # シグモイド基底関数
plt.xlabel('$x$')
plt.ylabel('$\phi_j(x)$')
plt.suptitle('Basis Function', fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%

# 基底関数の数を指定
M = 10

# パラメータを指定
mu_vals = np.linspace(-1.0, 1.0, M)
#mu_vals = np.repeat(0.0, M)
#s_vals = np.linspace(0.1, 1.0, M)
s_vals = np.repeat(0.1, M)

# M個の基底関数を作図
plt.figure(figsize=(12, 9))
for j in range(M):
#    plt.plot(x_vals, phi_poly(x_vals, j), label='j=' + str(j)) # 多項式基底関数
#    plt.plot(x_vals, phi_gauss(x_vals, mu_vals[j], s_vals[j]), 
#             label='$\mu=' + str(np.round(mu_vals[j], 1)) + ', s=' + str(np.round(s_vals[j], 1)) + '$') # ガウス基底関数
    plt.plot(x_vals, phi_sigmoid(x_vals, mu_vals[j], s_vals[j]), 
             label='$\mu=' + str(np.round(mu_vals[j], 1)) + ', s=' + str(np.round(s_vals[j], 1)) + '$') # シグモイド基底関数
plt.xlabel('$x$')
plt.ylabel('$\phi_j(x)$')
plt.suptitle('Basis Function', fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%

# フレーム数を指定
M = 101

# パラメータを指定
#mu_vals = np.linspace(-1.0, 1.0, M)
mu_vals = np.repeat(0.0, M)
s_vals = np.linspace(0.1, 1.0, M)
#s_vals = np.repeat(0.2, M)

# 図を初期化
fig = plt.figure(figsize=(8, 6))
fig.suptitle('Basis Function', fontsize=20)

# 作図処理を関数として定義
def update(j):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータを取得
    mu = mu_vals[j]
    s = s_vals[j]
    
    # 基底関数を作図
#    plt.plot(x_vals, phi_poly(x_vals, j), label='j=' + str(j)) # 多項式基底関数
    plt.plot(x_vals, phi_gauss(x_vals, mu, s), 
             label='$\mu=' + str(np.round(mu, 2)) + ', s=' + str(np.round(s, 2)) + '$') # ガウス基底関数
#    plt.plot(x_vals, phi_sigmoid(x_vals, mu, s), 
#             label='$\mu=' + str(np.round(mu, 2)) + ', s=' + str(np.round(s, 2)) + '$') # シグモイド基底関数
    plt.xlabel('$x$')
    plt.ylabel('$\phi_j(x)$')
    plt.legend(loc='upper left')
    plt.grid()
    plt.ylim(-0.1, 1.1)

# gif画像を作成
anime_basis = FuncAnimation(fig, update, frames=M, interval=50)

# gif画像を保存
anime_basis.save('PRML/Fig/ch3_1_0_Function1D.gif')


#%%

### 2次元の場合

## 基底関数の作成

# 2次元多項式基底関数を作成
def phi_poly2d(x, j):
    # 全ての次元の和をとる
    return np.sum(x**j, axis=1)

# 2次元ガウス基底関数を作成
def phi_gauss2d(x_d, mu_d, s_d):
    # 入力をパラメータで調整
    a_d = -(x_d - mu_d)**2 / (2.0 * s_d**2)
    
    # 全ての次元の和をとる
    s = np.sum(a_d, axis=1)
    return np.exp(s)

# 2次元シグモイド基底関数を作成
def phi_sigmoid2d(x_d, mu_d, s_d):
    # 入力をパラメータで調整
    a_d = (x_d - mu_d) / s_d
    
    # ロジスティックシグモイド関数の計算
    y_d = 1.0 / (1.0 + np.exp(-a_d))
    
    # 全ての次元の平均を計算
    return np.sum(y_d, axis=1) / 2.0

#%%

# 作図用のxの値を指定
x_vals = np.linspace(-1.0, 1.0, 201)

# 作図用のxの点を作成
X1, X2 = np.meshgrid(x_vals, x_vals)

# 計算用のxの点を作成
x_points = np.stack([X1.flatten(), X2.flatten()], axis=1)
x_dims = X1.shape
print(x_points.shape)
print(x_dims)

#%%

# パラメータを指定
j = 3
mu_d = np.array([0.0, 0.0])
s_d = np.array([0.2, 0.2])

# 2次元基底関数を計算
Z = phi_poly2d(x_points, j) # 多項式基底関数
#Z = phi_gauss2d(x_points, mu_d, s_d) # ガウス基底関数
#Z = phi_sigmoid2d(x_points, mu_d, s_d) # シグモイド基底関数

# 2次元基底関数を作図
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X1, X2, Z.reshape(x_dims), cmap='jet') # 曲面
ax.contour(X1, X2, Z.reshape(x_dims), cmap='jet', offset=np.min(Z)) # 等高線
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\phi_j(x)$')
fig.suptitle('Basis Function', fontsize=20)
ax.set_title('j=' + str(j), loc='left')
#ax.set_title('$\mu=(' + ', '.join([str(mu) for mu in mu_d]) + ')' + 
#             ', s=(' + ', '.join([str(s) for s in s_d]) + ')$', loc='left')
plt.show()

#%%

# フレーム数を指定
M = 101

# パラメータを指定
#mu_vals = np.stack([np.linspace(-1.0, 1.0, M), np.linspace(-1.0, 1.0, M)], axis=1)
mu_vals = np.stack([np.repeat(0.0, M), np.repeat(0.0, M)], axis=1)
s_vals = np.stack([np.linspace(0.1, 1.0, M), np.linspace(0.1, 1.0, M)], axis=1)
#s_vals = np.stack([np.repeat(0.1, M), np.repeat(0.1, M)], axis=1)

# 図を初期化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Basis Function', fontsize=20)

# 作図処理を関数として定義
def update(j):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # j番目のパラメータを取得
    mu_d = mu_vals[j]
    s_d = s_vals[j]
    
    # j番目の基底関数を計算
#    Z = phi_poly2d(x_points, j) # 多項式基底関数
#    Z = phi_gauss2d(x_points, mu_d, s_d) # ガウス基底関数
    Z = phi_sigmoid2d(x_points, mu_d, s_d) # シグモイド基底関数
    
    # 2次元多項式基底関数を作図
    ax.plot_surface(X1, X2, Z.reshape(x_dims), cmap='jet') # 曲面
    ax.contour(X1, X2, Z.reshape(x_dims), cmap='jet', offset=-0.1) # 等高線
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\phi_j(x)$')
#    ax.set_title('j=' + str(j), loc='left')
    ax.set_title('$\mu=(' + ', '.join([str(mu) for mu in np.round(mu_d, 2)]) + ')' + 
                 ', s=(' + ', '.join([str(s) for s in np.round(s_d, 2)]) + ')$', loc='left')
    ax.set_zlim(-0.1, 1.1)
    ax.view_init(elev=0, azim=315) # 表示アングル

# gif画像を作成
anime_basis = FuncAnimation(fig, update, frames=M, interval=50)

# gif画像を保存
anime_basis.save('PRML/Fig/ch3_1_0_BasisFunction2D.gif')


#%%

# 基底関数の数の平方根を指定
M_sqrt = 3

# √M個のパラメータの値を指定
mu_vals = np.linspace(-1.0, 1.0, M_sqrt)
s_vals = np.repeat(0.2, M_sqrt)

# 格子点を作成
Mu1, Mu2 = np.meshgrid(mu_vals, mu_vals)
S1, S2 = np.meshgrid(s_vals, s_vals)

# 計算用のパラメータを作成
mu_jd = np.stack([Mu1.flatten(), Mu2.flatten()], axis=1)
s_jd = np.stack([S1.flatten(), S2.flatten()], axis=1)

# M個の2次元ガウス基底関数を作図
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
for j in range(M_sqrt**2):
    # j番目のパラメータを取得
    mu_d = mu_jd[j]
    s_d = s_jd[j]
    
    # j番目の基底関数を計算
    Z = phi_gauss2d(x_points, mu_d, s_d)
    
    # 2次元ガウス基底関数を作図
    ax.plot_surface(X1, X2, Z.reshape(x_dims), cmap='jet', alpha=0.5) # 曲面
    ax.contour(X1, X2, Z.reshape(x_dims), cmap='jet', offset=0.0) # 等高線
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\phi_j(x)$')
ax.set_title('$s=(' + ', '.join([str(s) for s in np.round(s_d, 1)]) + ')$', loc='left')
fig.suptitle('Gaussian Basis Function', fontsize=20)
plt.show()


#%%

print('end')

