# 9.3.3 混合ベルヌーイ分布のEMアルゴリズム

#%%

# 9.3.3項で利用するライブラリ
import numpy as np
from scipy.stats import multinomial # 多項分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ゼロつく3のモジュールを利用
import sys, os
print(os.getcwd())
sys.path.append('../JupyterLab_Working/DL_from_Scratch3/deep-learning-from-scratch-3-master')
from dezero.datasets import MNIST

#%%

## MNISTデータセットの取得

# 訓練データを取得
train_set = MNIST(train=True)

# 入力データ・教師データを取得
train_x = train_set.data.reshape((60000, 28**2)) / 255 # 前処理
train_t = train_set.label
print(train_x.shape)
print(train_t.shape)

#%%

# 表示するデータ番号を指定
n = 0

# 手書き文字を表示:(256段階)
plt.imshow(train_x[n].reshape((28, 28)), cmap='gray')
#plt.imshow(train_x[n].reshape((1, 28**2)), cmap='gray')
plt.title('label:' + str(train_t[n]))
plt.axis('off') # 軸ラベル
plt.show()

#%%

## 利用する数字を指定する場合

# データ数を指定
N = 5000

# クラスを指定
label_list = [0, 2, 4, 6, 8]

# クラス数を設定
K = len(label_list)


# 指定したクラスのデータを抽出
x_arr = np.empty((0, 784))
t_arr = np.empty(0)
for k in range(K):
    # クラスkのデータを抽出
    x_arr = np.append(x_arr, train_x[train_t == label_list[k]], axis=0)
    t_arr = np.append(t_arr, train_t[train_t == label_list[k]], axis=0)

# 指定した数のデータをランダムに抽出
idx = np.random.permutation(len(x_arr))[:N] # 抽出するデータ番号
x_nd = x_arr[idx]
t_n = t_arr[idx].astype('int')
print(x_nd.shape)
print(t_n.shape)

#%%

## データ数の削減

# データ数を指定
N = 10000

# クラス数の設定:(固定)
K = 10

# シャッフルして指定した数のデータを抽出
idx = np.random.permutation(len(train_x))[:N]
x_nd = train_x[idx]
t_n = train_t[idx]
print(x_nd.shape)
print(t_n.shape)

#%%

# 2値に変換
x_nd = (x_nd > 0.5).astype('int')

#%%

# 表示するデータ番号を指定
n = 0

# 手書き文字を表示:(2値)
plt.imshow(x_nd[n].reshape((28, 28)), cmap='gray')
plt.title('label:' + str(t_n[n]))
plt.axis('off') # 軸ラベル
plt.show()

#%%

## 初期値の設定

# 次元数を設定:(固定)
D = 784

# モデルのパラメータを初期化
mu_kd = np.random.uniform(low=0.25, high=0.75, size=(K, D))

# 混合係数を初期化
pi_k = np.repeat(1.0 / K, K)

# 負担率の変数を作成
gamma_nk = np.zeros((N, K))


## 推論処理

# 試行回数を指定
MaxIter = 50

# 推移の確認用の受け皿を作成
trace_L_i = []
trace_mu_ikd = [mu_kd.copy()]
trace_pi_ik = [pi_k.copy()]
trace_gamma_ink = []

# 最尤推定
for i in range(MaxIter):
    
    # 負担率を計算:式(9.56)
    for n in range(N):
        for k in range(K):
            #prob_x_d = np.c_[1.0 - mu_kd[k], mu_kd[k]][np.arange(D), x_nd[n]] # D個のベルヌーイ分布
            prob_x_d = multinomial.pmf(x=np.c_[x_nd[n], 1 - x_nd[n]], n=1, p=np.c_[mu_kd[k], 1 - mu_kd[k]]) # (SciPy)
            gamma_nk[n, k] = pi_k[k] * np.prod(prob_x_d) # 分子
    gamma_nk /= np.sum(gamma_nk, axis=1, keepdims=True) # 正規化
    
    # 各クラスタとなるデータ数の期待値を計算:式(9.57)
    N_k = np.sum(gamma_nk, axis=0)
    
    # モデルのパラメータの最尤解を計算:式(9.59)
    mu_kd = np.dot(gamma_nk.T, x_nd) / N_k.reshape((K, 1))
    
    # 混合係数の最尤解を計算:式(9.60)
    pi_k = N_k / N
    
    # 不完全データ対数尤度を計算:式(9.51)
    prob_mix_n = np.zeros(N)
    for n in range(N):
        for k in range(K):
            #prob_x_d = np.c_[1.0 - mu_kd[k], mu_kd[k]][np.arange(D), x_nd[n]] # D個のベルヌーイ分布
            prob_x_d = multinomial.pmf(x=np.c_[x_nd[n], 1 - x_nd[n]], n=1, p=np.c_[mu_kd[k], 1 - mu_kd[k]]) # (SciPy)
            prob_mix_n[n] += pi_k[k] * np.prod(prob_x_d) # 混合ベルヌーイ分布
    L = np.sum(np.log(prob_mix_n))
    
    # i回目の結果を記録
    trace_L_i.append(L)
    trace_mu_ikd.append(mu_kd.copy())
    trace_pi_ik.append(pi_k.copy())
    trace_gamma_ink.append(gamma_nk.copy())
    
    # 動作確認
    print(str(i + 1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + '%)')

 #%%

## モデルのパラメータの確認

# 表示するクラス番号を指定
k = 0

# クラスkのパラメータを表示
plt.imshow(mu_kd[k].reshape((28, 28)), cmap='gray')
plt.title('k=' + str(k))
plt.axis('off') # 軸ラベル
plt.show()

#%%

# サブプロットの列数を指定(1行になるとエラーになる)
n_col = 4
n_row = K // (n_col + 1) + 1

# モデルのパラメータをまとめて表示
fig, axes = plt.subplots(n_row, n_col, figsize=(8, 7), constrained_layout=True)
for k in range(K):
    # サブプロットのインデックスを計算
    r = k // n_col
    c = k % n_col
    
    # クラスkのパラメータ
    axes[r, c].imshow(mu_kd[k].reshape((28, 28)), cmap='gray')
    axes[r, c].axis('off')
    axes[r, c].set_title('k=' + str(k))

# 余ったサブプロットを初期化
for c in range(c + 1, n_col):
    axes[r, c].axis('off')

# 最後のサブプロットにパラメータの平均値を表示
axes[r, n_col - 1].imshow(mu_kd.mean(axis=0).reshape((28, 28)), cmap='gray')
axes[r, n_col - 1].set_title('iter:' + str(MaxIter) + ', N=' + str(N) + '\n (mean)')

fig.suptitle('EM Algorithm', fontsize=20) # グラフタイトル
plt.show()

#%%

# 画像サイズを指定
fig, axes = plt.subplots(n_row, n_col, figsize=(8, 7), constrained_layout=True)

# 作図処理を関数として定義
def update(i):
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のパラメータをまとめて表示
    for k in range(K):
        # サブプロットのインデックスを計算
        r = k // n_col
        c = k % n_col
        
        # クラスkのパラメータ
        axes[r, c].imshow(trace_mu_ikd[i][k].reshape((28, 28)), cmap='gray')
        axes[r, c].axis('off')
    
    # 余ったサブプロットを初期化
    for c in range(c + 1, n_col):
        axes[r, c].cla()
        axes[r, c].axis('off')
    
    # 最後のサブプロットにパラメータの平均値を表示
    axes[r, n_col - 1].imshow(trace_mu_ikd[i].mean(axis=0).reshape((28, 28)), cmap='gray')
    axes[r, n_col - 1].set_title('mean')
    
    # 最初のサブプロットのタイトルに試行回数を表示
    axes[0, 0].set_title('iter:' + str(i) + ', N=' + str(N), loc='left')
    fig.suptitle('EM Algorithm', fontsize=20)

# gif画像を作成
params_anime = FuncAnimation(fig, update, frames=len(trace_mu_ikd), interval=100)
params_anime.save('ch9_3_3_params.gif')

#%%

## 学習の推移の確認

# 対数尤度の推移を作図
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(trace_L_i)), trace_L_i)
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('EM Algorithm', fontsize=20)
plt.title('Log Likelihood', loc='left')
plt.grid() # グリッド線
plt.show()

#%%

## パラメータの推移の確認

# 表示するクラス番号を指定
k = 0

# モデルのパラメータの推移を作図
plt.figure(figsize=(8, 6))
for d in range(D):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_mu_ikd)[:, k, d], 
             alpha=0.5, label='d=' + str(d))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('EM Algorithm', fontsize=20)
plt.title('$\mu_{' + str(k) + '}$', loc='left')
#plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# 混合係数の推移を作図
plt.figure(figsize=(8, 6))
for k in range(K):
    plt.plot(np.arange(MaxIter + 1), np.array(trace_pi_ik)[:, k], 
             alpha=1, label='k=' + str(k))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('EM Algorithm', fontsize=20)
plt.title('$\pi$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# 表示するデータ番号を指定
n = 0

# 負担率の推移を作図
plt.figure(figsize=(8, 6))
for k in range(K):
    plt.plot(np.arange(MaxIter), np.array(trace_gamma_ink)[:, n, k], 
             alpha=1, label='k=' + str(k))
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('EM Algorithm', fontsize=20)
plt.title('$\gamma(z_{' + str(n) + '})$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## おまけ：分類結果の確認

# 元の数字と分類先の関係をカウント
res_rate_kk = np.zeros((K, K))
for lbl_k in range(K):
    # k番目の数字のデータにおいて、分類された(確率が最大の)クラス番号を抽出
    #res_cls_idx = np.argmax(gamma_nk[t_n == lbl_k], axis=1)
    res_cls_idx = np.argmax(gamma_nk[t_n == label_list[lbl_k]], axis=1) # (数字を指定した場合)
    for cls_k in range(K):
        # k番目のクラスに分類された割合を計算
        res_rate_kk[cls_k, lbl_k] = np.mean(res_cls_idx == cls_k)

# 正解ラベルと分類クラスのヒートマップを作成
plt.figure(figsize=(8, 8))
plt.imshow(res_rate_kk)
plt.xlabel('label') # x軸ラベル
plt.ylabel('class') # y軸ラベル
plt.xticks(np.arange(K)) # x軸目盛
#plt.xticks(np.arange(K), label_list) # x軸目盛:(数字を指定した場合)
plt.yticks(np.arange(K)) # y軸目盛
plt.title('iter:' + str(i + 1) + ', N=' + str(N), loc='left')
plt.colorbar() # 確率値
plt.show()

#%%

# 画像サイズを指定
fig = plt.figure(figsize=(7, 7))

# 作図処理を関数として定義
def update(i):
    # i回目のカウント
    res_rate_kk = np.zeros((K, K))
    for lbl_k in range(K):
        # k番目の数字のデータにおいて、分類された(確率が最大の)クラス番号を抽出
        res_cls_idx = np.argmax(trace_gamma_ink[i][t_n == lbl_k], axis=1)
        #res_cls_idx = np.argmax(trace_gamma_ink[i][t_n == label_list[lbl_k]], axis=1) # (数字を指定した場合)
        for cls_k in range(K):
            # k番目のクラスに分類された割合を計算
            res_rate_kk[cls_k, lbl_k] = np.mean(res_cls_idx == cls_k)
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のヒートマップ
    plt.imshow(res_rate_kk)
    plt.xlabel('label') # x軸ラベル
    plt.ylabel('class') # y軸ラベル
    plt.xticks(np.arange(K)) # x軸目盛
    #plt.xticks(np.arange(K), label_list) # x軸目盛:(数字を指定した場合)
    plt.yticks(np.arange(K)) # y軸目盛
    plt.title('iter:' + str(i + 1) + ', N=' + str(N), loc='left')

# gif画像を作成
cls_anime = FuncAnimation(fig, update, frames=len(trace_gamma_ink), interval=100)
cls_anime.save('ch9_3_3_class.gif')

#%%

print('end')

