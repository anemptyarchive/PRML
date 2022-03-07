### ch4.3.4 多クラスロジスティック回帰：入力が2次元の場合

# 4.3.4項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal # 多次元ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%%

## 利用する関数の作成

# ロジスティッシグモイド関数を作成
def sigmoid(x):
    # ロジスティッシグモイド関数の計算
    y = 1.0 / (1.0 + np.exp(-x))
    return y

# ソフトマックス関数の実装
def softmax(x_k):
    # 最大値を引く:オーバーフロー対策
    x_k -= np.max(x_k, axis=-1, keepdims=True)
    
    # ソフトマックス関数の計算:(3.10)
    return np.exp(x_k) / np.sum(np.exp(x_k), axis=-1, keepdims=True)


## 基底関数の作成

# BasisFunctions.pyを参照

#%%

## データの生成

# クラス数を指定
K = 3

# クラス割り当て確率を指定
pi_k = np.array([0.4, 0.35, 0.25])

# データ生成用のK個の平均を指定
mu_kd = np.array(
    [[-2.5, 1.5], 
     [0.0, 2.0], 
     [2.5, 2.5]]
)

# データ生成用のK個の分散共分散行列を指定
sigma2_kdd = np.array(
    [[[0.25, 0.2], [0.2, 0.25]], 
     [[0.25, -0.2], [-0.2, 0.25]], 
     [[0.25, 0.2], [0.2, 0.25]]]
)


# データ数を指定
N = 250

# 真のクラスを生成
t_nk = np.random.multinomial(n=1, pvals=pi_k, size=N)

# 真のクラス番号を抽出
_, t_n = np.where(t_nk == 1)

# 真のクラスに従いデータを生成
x_nd = np.array(
    [np.random.multivariate_normal(mean=mu_kd[k], cov=sigma2_kdd[k], size=1).flatten() for k in t_n]
)


# 作図用のxの値を作成
x1 = np.linspace(np.min(x_nd[:, 0]) - 0.5, np.max(x_nd[:, 0]) + 0.5, num=250)
x2 = np.linspace(np.min(x_nd[:, 1]) - 0.5, np.max(x_nd[:, 1]) + 0.5, num=250)

# 作図用のxの点を作成
X1, X2 = np.meshgrid(x1, x2)

# 計算用のxの点を作成
x_points = np.stack([X1.flatten(), X2.flatten()], axis=1)
x_dims = X1.shape
print(x_points.shape)
print(x_dims)

# 混合ガウス分布を計算
model_density = 0.0
for k in range(K):
    # クラスkの確率密度を加算
    model_density += pi_k[k] * multivariate_normal.pdf(x=x_points, mean=mu_kd[k], cov=sigma2_kdd[k])


#%%

# 観測モデルを作図
plt.figure(figsize=(12, 9))
for k in range(K):
    k_idx = t_n == k
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='class ' + str(k+1)) # 各クラスの観測データ
plt.contour(X1, X2, model_density.reshape(x_dims)) # データ生成分布
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Observation Data', fontsize=20)
plt.title('$\pi=(' + ', '.join([str(pi) for pi in pi_k]) + ')' + 
          ', N=' + str(N) + '=(' + ', '.join([str(np.sum(t_n == k)) for k in range(K)]) + ')$', loc='left')
plt.colorbar(label='density')
plt.legend()
plt.grid()
plt.show()

#%%

## ロジスティック回帰

# 試行回数を指定
max_iter = 10000

# 基底関数の数を指定
M = 3

# 基底関数を指定
#phi = Phi_polynomial2d
#phi = Phi_gauss2d
phi = Phi_sigmoid2d

# 基底関数により入力値を変換
phi_x_nm = phi(x_nd, M)

# 学習率を指定
eta = 0.01

# 重みパラメータを初期化
w_mk = np.random.uniform(-10.0, 10.0, size=(M, K))

# 変数を初期化
nabla_E_mk = np.zeros((M, K))
h_mkmk = np.zeros((M*K, M*K))

# 推移の記録用の配列を作成
trace_w_arr = np.zeros((max_iter, M, K)) # 重みパラメータ
trace_E_list = np.zeros(max_iter) # 負の対数尤度

# ニュートン-ラフソン法による推定
for i in range(max_iter):
    # 重み付き和を計算
    a_nk = np.dot(phi_x_nm, w_mk)
    
    # ソフトマックス関数による変換
    y_nk = softmax(a_nk)
    
    # クラスごとに更新
    for k in range(K):
        # 勾配を計算
        nabla_E_mk[:, k] = np.dot(phi_x_nm.T, y_nk[:, [k]] - t_nk[:, [k]]).flatten()
        
        # 最急降下法によりパラメータを更新
        w_mk[:, k] -= eta * nabla_E_mk[:, k]
#        for j in range(K):
#            # 中間変数を計算
#            if k == j:
#                r_nn = np.diag(y_nk[:, k] * (1.0 - y_nk[:, j]) + 1e-7)
#            else:
#                r_nn = np.diag(-y_nk[:, k] * y_nk[:, j] + 1e-7)
#            
#            # ヘッセ行列のk,jブロックを計算
#            tmp_h_mm = phi_x_nm.T.dot(r_nn).dot(phi_x_nm)
#            
#            # ヘッセ行列を格納
#            h_mkmk[(k*M):(k*M+M), (j*M):(j*M+M)] = tmp_h_mm
#            
#    # ニュートン-ラフソン法によるパラメータを更新
#    w_mk -= np.dot(np.linalg.inv(h_mkmk), nabla_E_mk.reshape(-1, 1)).reshape(K, M).T
    
    # 負の対数尤度関数を計算
    E_val = -np.sum(t_nk * np.log(y_nk + 1e-7))
    
    # 値を記録
    trace_w_arr[i] = w_mk.copy()
    trace_E_list[i] = E_val
    print(E_val)

#%%

### 推定結果の可視化

## クラス分類

# K個の色を指定
color_list = ['royalblue', 'orange', 'darkturquoise']

# K個の回帰曲面を計算
phi_x_valsm = phi(x_points, M, x_nd)
a_valsk = np.dot(phi_x_valsm, w_mk)
y_valsk = softmax(a_valsk)

#%%

# K個の決定境界を作図
plt.figure(figsize=(12, 9))
plt.contour(X1, X2, model_density.reshape(x_dims)) # データ生成分布
for k in range(K):
    # クラスkの観測データを描画
    k_idx = t_n == k
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], 
                color=color_list[k], label='class ' + str(k+1))
    
    # クラスkの決定境界を描画
    plt.contour(X1, X2, y_valsk[:, k].reshape(x_dims), 
                colors=['white', color_list[k], 'white'], levels=[0.0, 0.5, 1.0])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.suptitle('Logistic Regression', fontsize=20)
plt.title('iter:' + str(max_iter) + ', N=' + str(N) + 
          ', E(w)=' + str(np.round(E_val, 3)), loc='left')
plt.colorbar(label='t')
plt.legend()
plt.grid()
plt.show()

#%%

# クラス番号を指定
j = 0

# 回帰曲面を作図
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
for k in range(K):
    k_idx = t_n == k
    if k == j:
        ax.scatter(xs=x_nd[k_idx, 0], ys=x_nd[k_idx, 1], zs=np.repeat(1.0, k_idx.sum()), 
                   c=color_list[k], label='class ' + str(k+1)) # クラスjの観測データ
        ax.scatter(xs=x_nd[k_idx, 0], ys=x_nd[k_idx, 1], zs=np.repeat(0.0, k_idx.sum()), 
                   facecolor='none', edgecolors=color_list[k], linestyles='--') # クラスjの観測データ:(底面)
    else:
        ax.scatter(xs=x_nd[k_idx, 0], ys=x_nd[k_idx, 1], zs=np.repeat(0.0, k_idx.sum()), 
                   c=color_list[k], label='class ' + str(k+1)) # クラスj以外の観測データ
ax.plot_surface(X1, X2, y_valsk[:, j].reshape(x_dims), cmap='jet', alpha=0.5) # クラスjの回帰曲面
cntr = plt.contour(X1, X2, y_valsk[:, j].reshape(x_dims), colors=['white', color_list[j], 'white'], levels=[0.0, 0.5, 1.0], offset=0.5) # クラスjの決定境界
plt.contour(X1, X2, y_valsk[:, j].reshape(x_dims), colors=color_list[j], alpha=0.5, linestyles='--', levels=[0.5], offset=0.0) # クラスjの決定境界:(底面)
plt.contour(X1, X2, model_density.reshape(x_dims), alpha=0.5, linestyles=':', offset=0.0) # データ生成分布:(底面)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$w_j^T \phi_j(x)$')
fig.suptitle('Logistic Regression', fontsize=20)
plt.title('$iter:' + str(max_iter) + ', N=' + str(N) + ', E(w)=' + str(np.round(E_val, 1)) + 
              ', w_j=(' + ', '.join([str(w) for w in np.round(w_mk[: , j], 1)]) + ')$', loc='left')
fig.colorbar(cntr, shrink=0.5, aspect=10, label='t')
ax.legend()
#ax.view_init(elev=90, azim=270) # 表示アングル
plt.show()


#%%

# フレーム数を指定
frame_num = 50

# 1フレーム当たりの試行回数を計算
iter_per_frame = max_iter // frame_num

# 図を初期化
fig = plt.figure(figsize=(12, 9))
fig.suptitle('Logistic Regression', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータを取得
    w_mk = trace_w_arr[i*iter_per_frame]
    E_val = trace_E_list[i*iter_per_frame]
    
    # K個の回帰曲面を計算
    a_valsk = np.dot(phi_x_valsm, w_mk)
    y_valsk = softmax(a_valsk)
    
    # 決定境界を作図
    plt.contour(X1, X2, model_density.reshape(x_dims)) # データ生成分布
    for k in range(K):
        # クラスkの観測データを描画
        k_idx = t_n == k
        plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], 
                    color=color_list[k], label='class ' + str(k + 1))
        
        # クラスkの決定境界を描画
        plt.contour(X1, X2, y_valsk[:, k].reshape(x_dims), 
                    colors=['white', color_list[k], 'white'], levels=[0.0, 0.5, 1.0])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('iter:' + str(i*iter_per_frame+1) + ', E(w)=' + str(np.round(E_val, 3)), loc='left')
    plt.legend()
    plt.grid()

# gif画像を作成
anime_logistic = FuncAnimation(fig, update, frames=frame_num, interval=100)

# gif画像を保存
anime_logistic.save('PRML/Fig/ch4_3_4_LogisticRegression_cntr.gif')

#%%

# フレーム数を指定
frame_num = 50

# 1フレーム当たりの試行回数を計算
iter_per_frame = max_iter // frame_num

# クラス番号を指定
j = 0

# 図を初期化
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Logistic Regression', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータを取得
    w_mk = trace_w_arr[i*iter_per_frame]
    E_val = trace_E_list[i*iter_per_frame]
    
    # K個の回帰曲面を計算
    a_valsk = np.dot(phi_x_valsm, w_mk)
    y_valsk = softmax(a_valsk)
    
    # 回帰曲面を作図
    for k in range(K):
        k_idx = t_n == k
        if k == j:
            ax.scatter(xs=x_nd[k_idx, 0], ys=x_nd[k_idx, 1], zs=np.repeat(1.0, k_idx.sum()), 
                       c=color_list[k], label='class ' + str(k+1)) # クラスjの観測データ
            ax.scatter(xs=x_nd[k_idx, 0], ys=x_nd[k_idx, 1], zs=np.repeat(0.0, k_idx.sum()), 
                       facecolor='none', edgecolors=color_list[k], linestyles='--') # クラスjの観測データ:(底面)
        else:
            ax.scatter(xs=x_nd[k_idx, 0], ys=x_nd[k_idx, 1], zs=np.repeat(0.0, k_idx.sum()), 
                       c=color_list[k], label='class ' + str(k+1)) # クラスj以外の観測データ
    ax.plot_surface(X1, X2, y_valsk[:, j].reshape(x_dims), cmap='jet', alpha=0.5) # クラスjの回帰曲面
    plt.contour(X1, X2, y_valsk[:, j].reshape(x_dims), colors=['white', color_list[j], 'white'], levels=[0.0, 0.5, 1.0], offset=0.5) # クラスjの決定境界
    plt.contour(X1, X2, y_valsk[:, j].reshape(x_dims), colors=color_list[j], alpha=0.5, linestyles='--', levels=[0.5], offset=0.0) # クラスjの決定境界:(底面)
    plt.contour(X1, X2, model_density.reshape(x_dims), alpha=0.5, linestyles=':', offset=0.0) # データ生成分布:(底面)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$w_j^T \phi_j(x)$')
    ax.set_title('$iter:' + str(i*iter_per_frame + 1) + ', E(w)=' + str(np.round(E_val, 1)) + 
                 ', w_j=(' + ', '.join([str(w) for w in np.round(w_mk[: , j], 1)]) + ')$', loc='left')
    ax.legend()
    #ax.view_init(elev=0, azim=315) # 表示アングル:(横から)
    #ax.view_init(elev=90, azim=270) # 表示アングル:(上から)

# gif画像を作成
anime_logistic = FuncAnimation(fig, update, frames=frame_num, interval=100)

# gif画像を保存
anime_logistic.save('PRML/Fig/ch4_3_4_LogisticRegression2D_srfc.gif')

#%%
