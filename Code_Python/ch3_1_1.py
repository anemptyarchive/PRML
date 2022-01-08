# ch3.1.1 線形基底関数モデル

# 3.1.1項で利用するライブラリ
import numpy as np
import matplotlib.pyplot as plt

#%%

## 基底関数の作成

# BasisFunctions.pyを参照

#%%

## モデルの設定

# 真の関数を作成
def y_true(x):
    # 計算式を指定
    return np.sin(2.0 * np.pi * x)

# 作図用のxの値を指定
x_vals = np.linspace(0.0, 0.1, num=101)

# 真の精度パラメータを指定
beta_true = 3.0

# 真の標準偏差を計算
sigma_true = np.sqrt(1.0 / beta_true)
sigma_true

# 真のモデルを計算
y_true_vals = y_true(x_vals)

#%%

# 真のモデルを作図
plt.figure(figsize=(12, 8))
plt.plot(x_vals, y_true_vals, color='darkturquoise', label='true model') # 真のモデル
plt.fill_between(x=x_vals, y1=y_true_vals-2.0*sigma_true, y2=y_true_vals+2.0*sigma_true, 
                 color='darkturquoise', alpha=0.2, linestyle='--') # 真のノイズ範囲
plt.xlabel('x')
plt.ylabel('t')
plt.suptitle('$t = \\sin(2 \pi x)$', fontsize=20)
plt.title('$\\beta=' + str(beta_true) + '$', loc='left')
plt.grid()
plt.legend()
plt.show()

#%%

## データの生成

# データ数を指定
N = 100

# データを生成
x_n = np.random.uniform(low=np.min(x_vals), high=np.max(x_vals), size=N)
t_n = y_true(x_n) + np.random.normal(loc=0.0, scale=sigma_true, size=N)

#%%

# 観測データの散布図を作成
plt.figure(figsize=(12, 8))
plt.plot(x_vals, y_true_vals, color='darkturquoise', label='true model') # 真のモデル
plt.fill_between(x=x_vals, y1=y_true_vals-2.0*sigma_true, y2=y_true_vals+2.0*sigma_true, 
                 color='darkturquoise', alpha=0.2, linestyle='--') # 真のノイズ範囲
plt.scatter(x_n, t_n, color='orange') # 観測データ
plt.xlabel('x')
plt.ylabel('t')
plt.suptitle('$t_n = \\sin(2 \pi x_n) + \epsilon_n$', fontsize=20)
plt.title('$N=' + str(N) + ', \\beta=' + str(beta_true) + '$', loc='left')
plt.grid()
plt.legend()
plt.show()

#%%

## 最尤推定

# 基底関数を指定
#phi = Phi_polynomial
#phi = Phi_gauss
phi = Phi_sigmoid

# 基底関数の数を指定
M = 6

# 基底関数により入力を変換
phi_x_nm = phi(x_n, M)

# 重みパラメータの最尤解を計算
w_ml_m = np.linalg.inv(np.dot(phi_x_nm.T, phi_x_nm)).dot(phi_x_nm.T).dot(t_n.reshape(-1, 1)).flatten()
print(w_ml_m)

# 分散パラメータの最尤解を計算
sigma2_ml = np.sum((t_n - np.dot(phi_x_nm, w_ml_m.reshape(-1, 1)).flatten())**2) / N
print(sigma2_ml)

# 精度パラメータの最尤解を計算
beta_ml = 1.0 / sigma2_ml
print(beta_ml)

#%%

## 推定結果の確認

# 作図用のxを変換
#phi_x_valsm = phi(x_vals, M) # 多項式基底関数
phi_x_valsm = phi(x_vals, M, x_n) # ガウス基底関数・シグモイド基底関数

# 回帰曲線を計算
y_ml_vals = np.dot(phi_x_valsm, w_ml_m.reshape(-1, 1)).flatten()

# 回帰曲線を作図
plt.figure(figsize=(12, 8))
plt.plot(x_vals, y_true_vals, color='darkturquoise', label='true model') # 真のモデル
plt.fill_between(x=x_vals, y1=y_true_vals-2.0*sigma_true, y2=y_true_vals+2.0*sigma_true, 
                 color='darkturquoise', alpha=0.2, linestyle=':') # 真のノイズ範囲
plt.plot(x_vals, y_ml_vals, color='blue', label='$w^T \phi(x)$') # 推定したモデル
plt.fill_between(x=x_vals, y1=y_ml_vals-2.0*np.sqrt(sigma2_ml), y2=y_ml_vals+2.0*np.sqrt(sigma2_ml), 
                 color='blue', alpha=0.2, linestyle=':') # 推定したノイズ範囲
plt.scatter(x_n, t_n, color='orange') # 観測データ
plt.xlabel('x')
plt.ylabel('t')
plt.suptitle('Linear Basis Function Model', fontsize=20)
plt.title('$N=' + str(N) + ', M=' + str(M) + 
          ', w=(' + ', '.join([str(w) for w in np.round(w_ml_m, 2)]) + ')' + 
          ', \\beta=' + str(np.round(beta_ml, 2)) + '$', loc='left')
plt.legend()
plt.grid()
plt.show()

#%%

# 基底関数を作図
plt.figure(figsize=(12, 8))
for m in range(M):
    plt.plot(x_vals, phi_x_valsm[:, m], label='$\phi_' + str(m) + '(x)$') # 基底関数
plt.xlabel('$x$')
plt.ylabel('$\phi_m(x)$')
plt.suptitle('Basis Function', fontsize=20)
plt.legend()
plt.grid()
plt.show()

#%%

# 重み付き基底関数を作図
plt.figure(figsize=(12, 8))
for m in range(M):
    plt.plot(x_vals, w_ml_m[m] * phi_x_valsm[:, m], 
             linestyle='--', label='$w_' + str(m) + '\phi_' + str(m) + '(x)$') # 重み付き基底関数
plt.plot(x_vals, y_ml_vals, color='blue', label='$w^T \phi(x)$')
plt.xlabel('x')
plt.ylabel('t')
plt.suptitle('Basis Function', fontsize=20)
plt.title('w=(' + ', '.join([str(w) for w in np.round(w_ml_m, 2)]) + ')', loc='left')
plt.legend()
plt.grid()
#plt.ylim(-5.1, 5.1)
plt.show()


#%%

print('end')

