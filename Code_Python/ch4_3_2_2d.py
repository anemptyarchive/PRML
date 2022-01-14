# ch4.3.2-3 ロジスティック回帰：入力が2次元の場合

# 4.3.2-3項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal # 多次元ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%

### モデルの設定

## 利用する関数の作成

# ロジスティックシグモイド関数を作成
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


## 基底関数の作成

# BasisFunctions.pyを参照

#%%

## データの生成

# クラス割り当て確率を指定:(クラス0, クラス1)
pi_k = np.array([0.4, 0.6])

# データ生成の平均を指定:(クラス0, クラス1)
mu_kd = np.array([[-2.5, 1.5], [0.0, 2.0]])

# データ生成の分散共分散行列を指定:(クラス0, クラス1)
sigma2_kdd = np.array(
    [[[0.25, 0.2], [0.2, 0.25]], 
     [[0.25, -0.2], [-0.2, 0.25]]]
)


# データ数を指定
N = 150

# 真のクラスを生成
t_n = np.random.binomial(n=1, p=pi_k[1], size=N)

# 真のクラスに従いデータを生成
x_nd = np.array(
    [np.random.multivariate_normal(mean=mu_kd[k], cov=sigma2_kdd[k], size=1).flatten() for k in t_n]
)


# 作図用のxの値を作成
x1 = np.linspace(np.min(x_nd[:, 0]) - 0.5, np.max(x_nd[:, 0]) + 0.5, num=100)
x2 = np.linspace(np.min(x_nd[:, 1]) - 0.5, np.max(x_nd[:, 1]) + 0.5, num=100)

# 作図用のxの点を作成
X1, X2 = np.meshgrid(x1, x2)

# 計算用のxの点を作成
x_points = np.stack([X1.flatten(), X2.flatten()], axis=1)
x_dims = X1.shape
print(x_points.shape)
print(x_dims)

# 混合ガウス分布を計算
model_density = 0.0
for k in range(len(pi_k)):
    # クラスkの確率密度を加算
    model_density += pi_k[k] * multivariate_normal.pdf(x=x_points, mean=mu_kd[k], cov=sigma2_kdd[k])

#%%

# 観測モデルを作図
plt.figure(figsize=(12, 9))
plt.scatter(x=x_nd[t_n == 0, 0], y=x_nd[t_n == 0, 1], color='darkturquoise', label='class 0') # クラス0の観測データ
plt.scatter(x=x_nd[t_n == 1, 0], y=x_nd[t_n == 1, 1], color='orange', label='class 1') # クラス1の観測データ
plt.contour(X1, X2, model_density.reshape(x_dims)) # データ生成分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$\pi=(' + ', '.join([str(pi) for pi in pi_k]) + ')' + 
          ', N_0=' + str(np.sum(t_n == 0)) + ', N_1=' + str(np.sum(t_n == 1)) + '$', loc='left')
plt.colorbar(label='density')
plt.legend()
plt.grid()
plt.show()

#%%

### ロジスティック回帰

# 試行回数を指定
max_iter = 100

# 基底関数の数を指定
M = 4

# 基底関数を指定
#phi = Phi_polynomial2d
#phi = Phi_gauss2d
phi = Phi_sigmoid2d

# 基底関数により入力値を変換
phi_x_nm = phi(x_nd, M)

# 重みパラメータを初期化
w_m = np.random.uniform(-10.0, 10.0, size=M)

# 推移の記録用の配列を作成
trace_w_arr = np.zeros((max_iter, M)) # 重みパラメータ
trace_E_list = np.zeros(max_iter) # 負の対数尤度

# ニュートン-ラフソン法による推定
for i in range(max_iter):
    # 重み付き和を計算
    a_n = np.dot(phi_x_nm, w_m.reshape(-1, 1)).flatten()
    
    # ロジスティックシグモイド関数による変換
    y_n = sigmoid(a_n)
    
    # 中間変数を計算
    r_nn = np.diag(y_n)
    z_n = np.dot(phi_x_nm, w_m.reshape(-1, 1)).flatten()
    z_n -= np.dot(np.linalg.inv(r_nn), (y_n - t_n).reshape(-1, 1)).flatten()
    w_term_mm = phi_x_nm.T.dot(r_nn).dot(phi_x_nm)
    w_term_m1 = phi_x_nm.T.dot(r_nn).dot(z_n.reshape(-1, 1))
    
    # パラメータを更新
    w_m = np.dot(np.linalg.inv(w_term_mm), w_term_m1).flatten()
    
    # 負の対数尤度関数を計算
    y_n = sigmoid(np.dot(phi_x_nm, w_m.reshape(-1, 1)).flatten())
    #E_val = -np.sum(t_n * np.log(y_n) + (1.0 - t_n) * np.log(1.0 - y_n))
    E_val = -np.sum(np.log(y_n**t_n) + np.log((1.0 - y_n)**(1.0 - t_n)))
    
    # 値を記録
    trace_w_arr[i] = w_m.copy()
    trace_E_list[i] = E_val
    
    # 途中経過を表示
    print(i + 1)

#%%

### 推定結果の可視化

## クラス分類

# 回帰曲面を計算
#phi_x_vals = phi(x_points, M) # 多項式基底関数の場合
phi_x_vals = phi(x_points, M, x_nd) # ガウス・シグモイド基底関数の場合
a_vals = np.dot(phi_x_vals, w_m).flatten()
y_vals = sigmoid(a_vals)

# 決定境界を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=x_nd[t_n == 0, 0], y=x_nd[t_n == 0, 1], color='darkturquoise', label='class 0') # クラス0の観測データ
plt.scatter(x=x_nd[t_n == 1, 0], y=x_nd[t_n == 1, 1], color='orange', label='class 1') # クラス1の観測データ
plt.contour(X1, X2, model_density.reshape(x_dims), alpha=0.5, linestyles='--') # データ生成分布
plt.contour(X1, X2, y_vals.reshape(x_dims), colors='red', levels=[0.0, 0.5, 1.0]) # 決定境界
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Logistic Regression', fontsize=20)
plt.title('iter:' + str(max_iter) + ', N=' + str(N) + 
          ', E(w)=' + str(np.round(E_val, 3)) + 
          ', w=(' + ', '.join([str(w) for w in np.round(w_m, 2)]) + ')', loc='left')
plt.colorbar(label='t')
plt.legend()
plt.grid()
plt.show()

#%%

# 回帰曲面を作図
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.scatter(x_nd[t_n == 0, 0], x_nd[t_n == 0, 1], np.repeat(0, np.sum(t_n == 0)), 
           c='darkturquoise', label='class 0') # クラス0の観測データ
ax.scatter(x_nd[t_n == 1, 0], x_nd[t_n == 1, 1], np.repeat(1, np.sum(t_n == 1)), 
           c='orange', label='class 1') # クラス1の観測データ
ax.scatter(x_nd[t_n == 0, 0], x_nd[t_n == 0, 1], np.repeat(0.0, np.sum(t_n == 0)), 
           facecolor='none', edgecolors='darkturquoise', linestyles='--') # クラス0の観測データ:(底面)
ax.scatter(x_nd[t_n == 1, 0], x_nd[t_n == 1, 1], np.repeat(0.0, np.sum(t_n == 1)), 
           facecolor='none', edgecolors='orange', linestyles='--') # クラス1の観測データ:(底面)
plt.contour(X1, X2, model_density.reshape(x_dims), alpha=0.5, linestyles=':', offset=0.0) # データ生成分布:(底面)
ax.plot_surface(X1, X2, y_vals.reshape(x_dims), cmap='jet', alpha=0.5) # 回帰曲面
surf = ax.contour(X1, X2, y_vals.reshape(x_dims), colors='red', levels=[0.0, 0.5, 1.0], offset=0.5) # 決定境界
ax.contour(X1, X2, y_vals.reshape(x_dims), colors='red', alpha=0.5, linestyles='--', levels=[0.5], offset=0.0) # 決定境界:(底面)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('t')
fig.suptitle('Logistic Regression', fontsize=20)
ax.set_title('iter:' + str(max_iter) + ', N=' + str(N) + 
          ', E(w)=' + str(np.round(E_val, 3)) + 
             ', w=(' + ', '.join([str(w) for w in np.round(w_m, 2)]) + ')', loc='left')
fig.colorbar(surf, shrink=0.5, aspect=10, label='t')
ax.legend()
#ax.view_init(elev=0, azim=315) # 表示アングル:(横から)
#ax.view_init(elev=90, azim=270) # 表示アングル:(上から)
plt.show()

#%%

# 図を初期化
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Logistic Regression', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータを取得
    w_m = trace_w_arr[i]
    E = trace_E_list[i]
    
    # 回帰曲面を計算
    a_vals = np.dot(phi_x_vals, w_m).flatten()
    y_vals = sigmoid(a_vals)
    
    # 決定境界を作図
    plt.scatter(x=x_nd[t_n == 0, 0], y=x_nd[t_n == 0, 1], c='darkturquoise', label='class 0') # クラス0の観測データ
    plt.scatter(x=x_nd[t_n == 1, 0], y=x_nd[t_n == 1, 1], c='orange', label='class 1') # クラス1の観測データ
    plt.contour(X1, X2, model_density.reshape(x_dims), alpha=0.5, linestyles='--') # データ生成分布
    plt.contour(X1, X2, y_vals.reshape(x_dims), colors='red', levels=[0.0, 0.5, 1.0]) # 決定境界
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('iter:' + str(i+1) + ', E(w)=' + str(np.round(E, 3)) + 
              ', w=(' + ', '.join([str(w) for w in np.round(w_m, 2)]) + ')', loc='left')
    plt.legend()
    plt.grid()

# gif画像を作成
anime_logistic = FuncAnimation(fig, update, frames=max_iter, interval=100)

# gif画像を保存
anime_logistic.save('PRML/Fig/ch4_3_2_LogisticRegression2D_cntr.gif')

#%%

# 図を初期化
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Logistic Regression', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータを取得
    w_m = trace_w_arr[i]
    E = trace_E_list[i]
    
    # 回帰曲面を計算
    a_vals = np.dot(phi_x_vals, w_m).flatten()
    y_vals = sigmoid(a_vals)
    
    # 回帰曲面を作図
    ax.scatter(x_nd[t_n == 0, 0], x_nd[t_n == 0, 1], np.repeat(0, np.sum(t_n == 0)), 
               c='darkturquoise', label='class 0') # クラス0の観測データ
    ax.scatter(x_nd[t_n == 1, 0], x_nd[t_n == 1, 1], np.repeat(1, np.sum(t_n == 1)), 
               c='orange', label='class 1') # クラス1の観測データ
    ax.scatter(x_nd[t_n == 0, 0], x_nd[t_n == 0, 1], np.repeat(0.0, np.sum(t_n == 0)), 
               facecolor='none', edgecolors='darkturquoise', linestyles='--') # クラス0の観測データ:(底面)
    ax.scatter(x_nd[t_n == 1, 0], x_nd[t_n == 1, 1], np.repeat(0.0, np.sum(t_n == 1)), 
               facecolor='none', edgecolors='orange', linestyles='--') # クラス1の観測データ:(底面)
    plt.contour(X1, X2, model_density.reshape(x_dims), alpha=0.5, linestyles=':', offset=0.0) # データ生成分布:(底面)
    ax.plot_surface(X1, X2, y_vals.reshape(x_dims), cmap='jet', alpha=0.5) # 回帰曲面
    ax.contour(X1, X2, y_vals.reshape(x_dims), colors='red', levels=[0.5], offset=0.5) # 決定境界
    ax.contour(X1, X2, y_vals.reshape(x_dims), colors='red', alpha=0.5, linestyles='--', levels=[0.5], offset=0.0) # 決定境界:(底面)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('t')
    ax.set_title('iter:' + str(i+1) + ', E(w)=' + str(np.round(E, 3)) + 
                 ', w=(' + ', '.join([str(w) for w in np.round(w_m, 2)]) + ')', loc='left')
    ax.legend()
    #ax.view_init(elev=0, azim=315) # 表示アングル:(横から)
    #ax.view_init(elev=90, azim=270) # 表示アングル:(上から)

# gif画像を作成
anime_logistic = FuncAnimation(fig, update, frames=max_iter, interval=100)

# gif画像を保存
anime_logistic.save('PRML/Fig/ch4_3_2_LogisticRegression2D_srfc.gif')

#%%

## 基底関数と重みパラメータの関係

# 基底関数ごとに重み付け
weight_phi_x_vals = w_m * phi_x_vals

# 重み付けした基底関数を作図
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
for m in range(M):
#    ax.plot_surface(X1, X2, phi_x_vals[:, m].reshape(x_dims), 
#                    cmap='jet', vmin=np.min(phi_x_vals), vmax=np.max(phi_x_vals), alpha=0.25) # 基底関数
    ax.plot_surface(X1, X2, weight_phi_x_vals[:, m].reshape(x_dims), 
                    cmap='jet', vmin=np.min(weight_phi_x_vals), vmax=np.max(weight_phi_x_vals), alpha=0.25) # 重み付け基底関数
ax.plot_surface(X1, X2, a_vals.reshape(x_dims), 
                cmap='viridis', vmin=np.min(weight_phi_x_vals), vmax=np.max(weight_phi_x_vals), alpha=0.5) # 重み付き和
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$\phi_j(x)$') # 基底関数
#ax.set_zlabel('$w_j \phi_j(x)$') # 重み付け基底関数
fig.suptitle('Basis Function', fontsize=20)
ax.set_title('w=(' + ', '.join([str(w) for w in np.round(w_m, 2)]) + ')', loc='left')
ax.view_init(elev=0, azim=315) # 表示アングル
plt.show()

#%%

## 推移の確認

# ch4_3_2_1d.pyを参照


#%%

print('end')

