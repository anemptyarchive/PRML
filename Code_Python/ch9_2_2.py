# 9.2.2 多次元ガウス分布のEMアルゴリズム

#%%

# 9.2.2項で利用するライブラリ
import numpy as np
from scipy.stats import multivariate_normal # 多次元ガウス分布
import matplotlib.pyplot as plt

#%%

## 真の分布の設定

# 次元数を設定:(固定)
D = 2

# クラスタ数を指定
K = 3

# K個の真の平均を指定
mu_truth_kd = np.array(
    [[5.0, 35.0], 
     [-20.0, -10.0], 
     [30.0, -20.0]]
)

# K個の真の共分散行列を指定
sigma2_truth_kdd = np.array(
    [[[250.0, 65.0], [65.0, 270.0]], 
     [[125.0, -45.0], [-45.0, 175.0]], 
     [[210.0, -15.0], [-15.0, 250.0]]]
)

# 真の混合係数を指定
pi_truth_k = np.array([0.45, 0.25, 0.3])

#%%

# 作図用のx軸のxの値を作成
x_1_line = np.linspace(
    np.min(mu_truth_kd[:, 0] - 3 * np.sqrt(sigma2_truth_kdd[:, 0, 0])), 
    np.max(mu_truth_kd[:, 0] + 3 * np.sqrt(sigma2_truth_kdd[:, 0, 0])), 
    num=300
)

# 作図用のy軸のxの値を作成
x_2_line = np.linspace(
    np.min(mu_truth_kd[:, 1] - 3 * np.sqrt(sigma2_truth_kdd[:, 1, 1])), 
    np.max(mu_truth_kd[:, 1] + 3 * np.sqrt(sigma2_truth_kdd[:, 1, 1])), 
    num=300
)

# 作図用の格子状の点を作成
x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)

# 作図用のxの点を作成
x_point_arr = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)

# 作図用に各次元の要素数を保存
x_dim = x_1_grid.shape
print(x_dim)


# 真の分布を計算
model_dens = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_dens = multivariate_normal.pdf(
        x=x_point_arr, mean=mu_truth_kd[k], cov=sigma2_truth_kdd[k]
    )
    
    # K個の分布を線形結合
    model_dens += pi_truth_k[k] * tmp_dens

#%%

# 真の分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, model_dens.reshape(x_dim)) # 真の分布
plt.suptitle('Mixture of Gaussians', fontsize=20)
plt.title('K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar() # 等高線の色
plt.show()

#%%

# (観測)データ数を指定
N = 250

# 潜在変数を生成
z_truth_nk = np.random.multinomial(n=1, pvals=pi_truth_k, size=N)

# クラスタ番号を抽出
_, z_truth_n = np.where(z_truth_nk == 1)

# (観測)データを生成
x_nd = np.array([
    np.random.multivariate_normal(
        mean=mu_truth_kd[k], cov=sigma2_truth_kdd[k], size=1
    ).flatten() for k in z_truth_n
])

#%%

# 観測データの散布図を作成
plt.figure(figsize=(12, 9))
for k in range(K):
    k_idx, = np.where(z_truth_n == k) # クラスタkのデータのインデック
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], label='cluster:' + str(k + 1)) # 観測データ
plt.contour(x_1_grid, x_2_grid, model_dens.reshape(x_dim), linestyles='--') # 真の分布
plt.suptitle('Mixture of Gaussians', fontsize=20)
plt.title('$N=' + str(N) + ', K=' + str(K) + 
          ', \pi=(' + ', '.join([str(pi) for pi in pi_truth_k]) + ')$', loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar() # 等高線の色
plt.show()

#%%

## 初期値の設定

# 平均パラメータの初期値を生成
mu_kd = np.array([
    np.random.uniform(
        low=np.min(x_nd[:, d]), high=np.max(x_nd[:, d]), size=K
    ) for d in range(D)
]).T

# 共分散行列の初期値を指定
sigma2_kdd = np.array([np.identity(D) * 1000 for _ in range(K)])

# 混合係数の初期値を生成
pi_k = np.random.rand(3)
pi_k /= np.sum(pi_k) # 正規化


# 初期値による対数尤度を計算:式(9.14)
term_dens_nk = np.array(
    [multivariate_normal.pdf(x=x_nd, mean=mu_kd[k], cov=sigma2_kdd[k]) for k in range(K)]
).T
L = np.sum(np.log(np.sum(term_dens_nk, axis=1)))

#%%

# 初期値による混合分布を計算
init_dens = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_dens = multivariate_normal.pdf(
        x=x_point_arr, mean=mu_kd[k], cov=sigma2_kdd[k]
    )
    
    # K個の分布線形結合
    init_dens += pi_k[k] * tmp_dens

# 初期値による分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, model_dens.reshape(x_dim), 
            alpha=0.5, linestyles='dashed') # 真の分布
#plt.contour(x_1_grid, x_2_grid, init_dens.reshape(x_dim)) # 推定値による分布:(等高線)
plt.contourf(x_1_grid, x_2_grid, init_dens.reshape(x_dim), alpha=0.6) # 推定値による分布:(塗りつぶし)
plt.suptitle('Mixture of Gaussians', fontsize=20)
plt.title('iter:' + str(0) + ', K=' + str(K), loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.colorbar() # 等高線の色
plt.show()

#%%

## 推論処理

# 試行回数を指定
MaxIter = 100

# 推移の確認用の受け皿を作成
trace_L_i = [L]
trace_gamma_ink = [np.tile(np.nan, reps=(N, K))]
trace_mu_ikd = [mu_kd.copy()]
trace_sigma2_ikdd = [sigma2_kdd.copy()]
trace_pi_ik = [pi_k.copy()]

# 最尤推定
for i in range(MaxIter):
    
    # 負担率を計算:式(9.13)
    gamma_nk = np.array(
        [multivariate_normal.pdf(x=x_nd, mean=mu_kd[k], cov=sigma2_kdd[k]) for k in range(K)]
    ).T
    gamma_nk /= np.sum(gamma_nk, axis=1, keepdims=True) # 正規化
    
    # 各クラスタとなるデータ数の期待値を計算:式(9.18)
    N_k = np.sum(gamma_nk, axis=0)
    
    for k in range(K):
        # 平均パラメータの最尤解を計算:式(9.17)
        mu_kd[k] = np.dot(gamma_nk[:, k], x_nd) / N_k[k]
        
        # 共分散行列の最尤解を計算:式(9.19)
        term_x_nd = x_nd - mu_kd[k]
        sigma2_kdd[k] = np.dot(gamma_nk[:, k] * term_x_nd.T, term_x_nd) / N_k[k]
    
    # 混合係数の最尤解を計算:式(9.22)
    pi_k = N_k / N
    
    # 対数尤度を計算:式(9.14)
    tmp_dens_nk = np.array(
        [multivariate_normal.pdf(x=x_nd, mean=mu_kd[k], cov=sigma2_kdd[k]) for k in range(K)]
    ).T
    L = np.sum(np.log(np.sum(tmp_dens_nk, axis=1)))
    
    # i回目の結果を記録
    trace_L_i.append(L)
    trace_gamma_ink.append(gamma_nk.copy())
    trace_mu_ikd.append(mu_kd.copy())
    trace_sigma2_ikdd.append(sigma2_kdd.copy())
    trace_pi_ik.append(pi_k.copy())
    
    # 動作確認
    print(str(i + 1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

#%%

## 推論結果の確認

# K個のカラーマップを指定
colormap_list = ['Blues', 'Oranges', 'Greens']

# 負担率が最大のクラスタ番号を抽出
z_n = np.argmax(gamma_nk, axis=1)

# 割り当てられたクラスタとなる負担率(確率)を抽出
prob_z_n = gamma_nk[np.arange(N), z_n]

# 最後の更新値による混合分布を計算
res_dens = 0
for k in range(K):
    # クラスタkの分布の確率密度を計算
    tmp_dens = multivariate_normal.pdf(
        x=x_point_arr, mean=mu_kd[k], cov=sigma2_kdd[k]
    )
    
    # K個の分布を線形結合
    res_dens += pi_k[k] * tmp_dens

# 最後の更新値による分布を作図
plt.figure(figsize=(12, 9))
plt.contour(x_1_grid, x_2_grid, model_dens.reshape(x_dim), 
            alpha=0.5, linestyles='dashed') # 真の分布
plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], 
            color='red', s=100, marker='x') # 真の平均
plt.contourf(x_1_grid, x_2_grid, res_dens.reshape(x_dim), alpha=0.5) # 推定値による分布:(塗りつぶし)
for k in range(K):
    k_idx, = np.where(z_n == k) # クラスタkのデータのインデックス
    cm = plt.get_cmap(colormap_list[k]) # クラスタkのカラーマップを設定
    plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], 
                c=[cm(p) for p in prob_z_n[k_idx]], label='cluster:' + str(k + 1)) # 負担率によるクラスタ
#plt.contour(x_1_grid, x_2_grid, res_dens.reshape(x_dim)) # 推定値による分布:(等高線)
#plt.colorbar() # 等高線の値:(等高線用)
plt.suptitle('Mixture of Gaussians:Maximum Likelihood', fontsize=20)
plt.title('$iter:' + str(MaxIter) + 
          ', L=' + str(np.round(L, 1)) + 
          ', N=' + str(N) + 
          ', \pi=[' + ', '.join([str(pi) for pi in np.round(pi_k, 3)]) + ']$', 
          loc='left')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()

#%%

## 対数尤度の推移を作図
plt.figure(figsize=(12, 9))
plt.plot(np.arange(MaxIter + 1), np.array(trace_L_i))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Maximum Likelihood', fontsize=20)
plt.title('Log Likelihood', loc='left')
plt.grid() # グリッド線
plt.show()

#%%

## パラメータの更新値の推移の確認

# muの推移を作図
plt.figure(figsize=(12, 9))
plt.hlines(y=mu_truth_kd, xmin=0, xmax=MaxIter + 1, label='true val', 
           color='red', linestyles='--') # 真の値
for k in range(K):
    for d in range(D):
        plt.plot(np.arange(MaxIter+1), np.array(trace_mu_ikd)[:, k, d], 
                 label='k=' + str(k + 1) + ', d=' + str(d + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Maximum Likelihood', fontsize=20)
plt.title('$\mu$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# lambdaの推移を作図
plt.figure(figsize=(12, 9))
plt.hlines(y=sigma2_truth_kdd, xmin=0, xmax=MaxIter + 1, label='true val', 
           color='red', linestyles='--') # 真の値
for k in range(K):
    for d1 in range(D):
        for d2 in range(D):
            plt.plot(np.arange(MaxIter + 1), np.array(trace_sigma2_ikdd)[:, k, d1, d2], 
                     alpha=0.5, label='k=' + str(k + 1) + ', d=' + str(d1 + 1) + ', d''=' + str(d2 + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Maximum Likelihood', fontsize=20)
plt.title('$\Sigma$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# piの推移を作図
plt.figure(figsize=(12, 9))
plt.hlines(y=pi_truth_k, xmin=0, xmax=MaxIter + 1, label='true val', 
           color='red', linestyles='--') # 真の値
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_pi_ik)[:, k], label='k=' + str(k + 1))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Maximum Likelihood', fontsize=20)
plt.title('$\pi$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## アニメーションによる確認

# 追加ライブラリ
import matplotlib.animation as animation

#%%

## 分布の推移の確認

# K個のカラーマップを指定
colormap_list = ['Blues', 'Oranges', 'Greens']

# 画像サイズを指定
fig = plt.figure(figsize=(9, 9))

# 作図処理を関数として定義
def update_model(i):
    # i回目の更新値による混合分布を計算
    res_dens = 0
    for k in range(K):
        # クラスタkの分布の確率密度を計算
        tmp_dens = multivariate_normal.pdf(
            x=x_point_arr, mean=trace_mu_ikd[i][k], cov=trace_sigma2_ikdd[i][k]
        )
        
        # K個の分布を線形結合
        res_dens += trace_pi_ik[i][k] * tmp_dens
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目の負担率を取得
    gamma_nk = trace_gamma_ink[i]
    
    # i回目のサンプルによる分布を作図
    plt.contour(x_1_grid, x_2_grid, model_dens.reshape(x_dim), 
                alpha=0.5, linestyles='dashed') # 真の分布
    #plt.contour(x_1_grid, x_2_grid, res_dens.reshape(x_dim)) # 推定値による分布:(等高線)
    plt.contourf(x_1_grid, x_2_grid, res_dens.reshape(x_dim), alpha=0.5) # 推定値による分布:(塗りつぶし)
    plt.scatter(x=mu_truth_kd[:, 0], y=mu_truth_kd[:, 1], 
                color='red', s=100, marker='x') # 真の平均
    for k in range(K):
        k_idx, = np.where(np.argmax(gamma_nk, axis=1) == k)
        cm = plt.get_cmap(colormap_list[k]) # クラスタkのカラーマップを設定
        plt.scatter(x=x_nd[k_idx, 0], y=x_nd[k_idx, 1], 
                    c=[cm(p) for p in gamma_nk[k_idx, k]], label='cluster:' + str(k + 1)) # 負担率によるクラスタ
        plt.suptitle('Mixture of Gaussians:Maximum Likelihood', fontsize=20)
    plt.title('$iter:' + str(i) + 
              ', L=' + str(np.round(trace_L_i[i], 1)) + 
              ', N=' + str(N) + 
              ', \pi=(' + ', '.join([str(pi) for pi in np.round(trace_pi_ik[i], 3)]) + ')$', 
              loc='left')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

# gif画像を作成
model_anime = animation.FuncAnimation(fig, update_model, frames=MaxIter + 1, interval=100)
model_anime.save("ch9_2_2_Model.gif")


#%%

print('end')

