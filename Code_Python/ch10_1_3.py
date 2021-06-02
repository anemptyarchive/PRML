# 10.1.3 一変数ガウス分布の変分推論

#%%

# 10.1.3項で利用するライブラリ
import numpy as np
from scipy.stats import norm, gamma # 1次元ガウス分布, ガンマ分布
import matplotlib.pyplot as plt

#%%

## 真の分布(1次元ガウス分布)の設定

# 真の平均パラメータを指定
mu_truth = 5.0

# 真の精度パラメータを指定
tau_truth = 0.5
print(np.sqrt(1.0 / tau_truth)) # 標準偏差


# 作図用のxの値を作成
x_line = np.linspace(
    mu_truth - 4.0 * np.sqrt(1.0 / tau_truth), 
    mu_truth + 4.0 * np.sqrt(1.0 / tau_truth), 
    num=1000
)

# 真の分布を計算
model_dens = norm.pdf(x=x_line, loc=mu_truth, scale=np.sqrt(1.0 / tau_truth))

#%%

# 真の分布を作図
plt.figure(figsize=(12, 9))
plt.plot(x_line, model_dens, label='true model') # 真の分布
plt.xlabel('x')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$\mu=' + str(mu_truth) + ', \\tau=' + str(tau_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 観測データの生成

# (観測)データ数を指定
N = 50

# ガウス分布に従うデータを生成
x_n = np.random.normal(loc=mu_truth, scale=np.sqrt(1 / tau_truth), size=N)

#%%

# 観測データのヒストグラムを作図
plt.figure(figsize=(12, 9))
#plt.hist(x=x_n, bins=50, label='data') # 観測データ:(度数)
plt.hist(x=x_n, density=True, bins=50, label='data') # 観測データ:(相対度数)
plt.plot(x_line, model_dens, color='red', linestyle='--', label='true model') # 真の分布
plt.xlabel('x')
#plt.ylabel('count') # (度数用)
plt.ylabel('density') # (相対度数用)
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + ', \mu=' + str(mu_truth) + ', \\tau=' + str(tau_truth) + '$', loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 事前分布の設定

#　muの事前分布のパラメータを指定
mu_0 = 0.0
lambda_0 = 0.1

# tauの事前分布のパラメータを指定
a_0 = 1.0
b_0 = 1.0


# 作図用のmuの値を作成
mu_line = np.linspace(
    mu_truth - 4.0 * np.sqrt(1.0 / tau_truth), 
    mu_truth + 4.0 * np.sqrt(1.0 / tau_truth), 
    num=500
)

# 作図用のtauの値を作成
tau_line = np.linspace(0.0, 4 * tau_truth, num=500)

# 格子状の点を作成
mu_grid, tau_grid = np.meshgrid(mu_line, tau_line)

# 配列の形状を保存
point_dims = mu_grid.shape
print(point_dims)


# muの事前分布を計算
mu_prior_dens = norm.pdf(
    x=mu_grid.flatten(), loc=mu_0, scale=np.sqrt(1.0 / (lambda_0 * tau_grid.flatten() + 1e-7))
)

# tauの事前分布を計算
tau_prior_dens = gamma.pdf(x=tau_grid.flatten(), a=a_0, scale=1.0 / b_0)

# 同時事前分布を計算
prior_dens = mu_prior_dens * tau_prior_dens

#%%

# 事前分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth, y=tau_truth, color='red', s=100, marker='x', label='true val') # 真の値
plt.contour(mu_grid, tau_grid, prior_dens.reshape(point_dims)) # 事前分布
plt.xlabel('$\mu$')
plt.ylabel('$\\tau$')
plt.suptitle('Gaussian-Gamma Distribution', fontsize=20)
plt.title('$\mu_0=' + str(mu_0)+ ', \lambda_0=' + str(lambda_0) + 
          ', a_0=' + str(a_0) + ', b_0=' + str(b_0) + '$', 
          loc='left')
plt.colorbar() # 等高線の値
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 真の事後分布の計算

# muの真の事後分布のパラメータを計算
lambda_hat = lambda_0 + N
mu_hat = (lambda_0 * mu_0 + sum(x_n)) / lambda_hat

# lambdaの真の事後分布のパラメータを計算
a_hat = a_0 + 0.5 * N
b_hat = b_0 + 0.5 * (sum(x_n**2) + lambda_0 * mu_0**2 - lambda_hat * mu_hat**2)


# muの真の事後分布を計算
mu_true_posterior_dens = norm.pdf(
    x=mu_grid.flatten(), loc=mu_hat, scale=np.sqrt(1.0 / (lambda_hat * tau_grid.flatten() + 1e-7))
)

# tauの真の事後分布を計算
tau_true_posterior_dens = gamma.pdf(x=tau_grid.flatten(), a=a_hat, scale=1.0 / b_hat)

# 真の同時事後分布を計算
posterior_truth_dens = mu_true_posterior_dens * tau_true_posterior_dens

#%%

# 真の事後分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth, y=tau_truth, color='red', s=100, marker='x', label='true val') # 真の値
plt.contour(mu_grid, tau_grid, posterior_truth_dens.reshape(point_dims)) # 真の事後分布
plt.xlabel('$\mu$')
plt.ylabel('$\\tau$')
plt.suptitle('Gaussian-Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{\mu}=' + str(np.round(mu_hat, 1))+ ', \hat{\lambda}=' + str(np.round(lambda_hat, 5)) + 
          ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(np.round(b_hat, 1)) + '$', 
          loc='left')
plt.xlim(mu_truth - np.sqrt(1.0 / tau_truth), mu_truth + np.sqrt(1.0 / tau_truth))
plt.ylim(0.0, 2.0 * tau_truth)
plt.colorbar() # 等高線の値
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# muの真の事後分布を作図
mu_true_posterior_dens = norm.pdf(
    x=mu_line, loc=mu_hat, scale=np.sqrt(1.0 / (lambda_hat * a_hat / b_hat))
)

# muの真の事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(mu_line, mu_true_posterior_dens, label='$\mu$ posterior') # muの事後分布
plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(mu_true_posterior_dens), 
           color='red', linestyle='--', label='true val') # muの真の値
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{\mu}=' + str(np.round(mu_hat, 1)) + 
          ', \hat{\\lambda}=' + str(lambda_hat) + 
          ', E[\\tau]=' + str(np.round(a_hat / b_hat, 5)) + '$', 
          loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# tauの真の事後分布を計算
tau_true_posterior_dens = gamma.pdf(x=tau_line, a=a_hat, scale=1.0 / b_hat)

# lambdaの真の事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(tau_line, tau_true_posterior_dens, label='$\\tau$ posterior') # tauの事後分布
plt.vlines(x=tau_truth, ymin=0.0, ymax=np.nanmax(tau_true_posterior_dens), 
           color='red', linestyle='--', label='true val') # tauの真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \hat{a}=' + str(a_hat) + ', \hat{b}=' + str(np.round(b_hat, 1)) + '$', 
          loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 推論処理

# 試行回数を指定
MaxIter = 5

# 初期値を代入
mu_N = mu_0
lambda_N = lambda_0 * a_0 / a_0
a_N = a_0
b_N = b_0

# 推移の確認用の受け皿を作成
trace_mu_i = [mu_0]
trace_lambda_i = [lambda_N]
trace_a_i = [a_N]
trace_b_i = [b_N]

# 変分推論
for i in range(MaxIter):
    
    # mu の近似事後分布のパラメータを計算: 式 (10.26)(10.27)
    mu_N = (lambda_0 * mu_0 + np.sum(x_n)) / (lambda_0 + N)
    lambda_N = (lambda_0 + N) * a_N / b_N
    
    # i回目のmuの近似事後分布の更新後の結果を記録
    trace_mu_i.append(mu_N)
    trace_lambda_i.append(lambda_N)
    trace_a_i.append(a_N)
    trace_b_i.append(b_N)
    
    # tauの近似事後分布のパラメータを計算:式(10.29)(10.30)
    a_N = a_0 + 0.5 * (N + 1)
    b_N = b_0 + 0.5 * (lambda_0 * mu_0**2 + np.sum(x_n**2))
    b_N += 0.5 * (lambda_0 + N) * (mu_N**2 + 1.0 / lambda_N)
    b_N -= (lambda_0 * mu_0 + np.sum(x_n)) * mu_N
    
    # i回目のtauの近似事後分布の更新後の結果を記録
    trace_mu_i.append(mu_N)
    trace_lambda_i.append(lambda_N)
    trace_a_i.append(a_N)
    trace_b_i.append(b_N)
    
    # 動作確認
    print(str(i + 1) + ' (' + str(np.round((i + 1) / MaxIter * 100, 1)) + ')%')

#%%

## 推論結果の確認

# muの近似事後分布を計算
E_tau = a_N / b_N # tauの期待値
mu_posterior_dens = norm.pdf(
    x=mu_grid.flatten(), 
    loc=mu_N, 
    scale=np.sqrt(1.0 / (lambda_N / E_tau * tau_grid.flatten() + 1e-7))
)

# tauの近似事後分布を計算
tau_posterior_dens = gamma.pdf(x=tau_grid.flatten(), a=a_N, scale=1.0 / b_N)

# 同時近似事後分布を計算
posterior_dens = mu_posterior_dens * tau_posterior_dens

#%%

# 近似事後分布を作図
plt.figure(figsize=(12, 9))
plt.scatter(x=mu_truth, y=tau_truth, color='red', s=100, marker='x', label='true val') # 真の値
plt.contour(mu_grid, tau_grid, posterior_truth_dens.reshape(point_dims), 
            alpha=0.5, linestyles='--') # 真の事後分布
plt.contour(mu_grid, tau_grid, posterior_dens.reshape(point_dims)) # 近似事後分布
plt.xlabel('$\mu$')
plt.ylabel('$\\tau$')
plt.suptitle('Gaussian-Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \mu_N=' + str(np.round(mu_N, 1))+ 
          ', \lambda_N\ /\ E[\\tau]=' + str(np.round(lambda_N / E_tau, 1)) + 
          ', a_N=' + str(a_N) + 
          ', b_N=' + str(np.round(b_N, 1)) + '$', 
          loc='left')
plt.xlim(mu_truth - np.sqrt(1.0 / tau_truth), mu_truth + np.sqrt(1.0 / tau_truth))
plt.ylim(0.0, 2.0 * tau_truth)
plt.colorbar() # 等高線の値
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# muの近似事後分布を作図
mu_posterior_dens = norm.pdf(
    x=mu_line, loc=mu_N, scale=np.sqrt(1.0 / lambda_N)
)

# muの近似事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(mu_line, mu_posterior_dens, label='$\mu$ posterior') # muの近似事後分布
plt.vlines(x=mu_truth, ymin=0.0, ymax=np.nanmax(mu_posterior_dens), 
           color='red', linestyle='--', label='true val') # muの真の値
plt.xlabel('$\mu$')
plt.ylabel('density')
plt.suptitle('Gaussian Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', \mu_N=' + str(np.round(mu_N, 1)) + 
          ', \\lambda_N=' + str(np.round(lambda_N, 5)) + '$', 
          loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

# tauの近似事後分布を計算
tau_posterior_dens = gamma.pdf(x=tau_line, a=a_N, scale=1.0 / b_N)

# lambdaの近似事後分布を作図
plt.figure(figsize=(12, 9))
plt.plot(tau_line, tau_posterior_dens, label='$\\tau$ posterior') # tauの事後分布
plt.vlines(x=tau_truth, ymin=0.0, ymax=np.nanmax(tau_posterior_dens), 
           color='red', linestyle='--', label='true val') # tauの真の値
plt.xlabel('$\lambda$')
plt.ylabel('density')
plt.suptitle('Gamma Distribution', fontsize=20)
plt.title('$N=' + str(N) + 
          ', a_N=' + str(a_hat) + ', b_N=' + str(np.round(b_N, 1)) + '$', 
          loc='left')
plt.legend() # 凡例
plt.grid() # グリッド線
plt.show()

#%%

## 超パラメータの推移の確認

# mu_Nの推移を作図
plt.figure(figsize=(12, 9))
plt.plot(np.arange(0.0, MaxIter + 0.1, 0.5), trace_mu_i)
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\mu_N$', loc='left')
plt.grid() # グリッド線
plt.show()

#%%

# lambda_Nの推移を作図
plt.figure(figsize=(12, 9))
plt.plot(np.arange(0.0, MaxIter + 0.1, 0.5), trace_lambda_i)
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$\lambda_N$', loc='left')
plt.grid() # グリッド線
plt.show()

#%%

# a_Nの推移を作図
plt.figure(figsize=(12, 9))
plt.plot(np.arange(0.0, MaxIter + 0.1, 0.5), trace_a_i)
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$a_N$', loc='left')
plt.grid() # グリッド線
plt.show()

#%%

# b_Nの推移を作図
plt.figure(figsize=(12, 9))
plt.plot(np.arange(0.0, MaxIter + 0.1, 0.5), trace_b_i)
plt.xlabel('iteration')
plt.ylabel('value')
plt.suptitle('Variational Inference', fontsize=20)
plt.title('$b_N$', loc='left')
plt.grid() # グリッド線
plt.show()

#%%

## アニメーションによる推移の確認

# 追加ライブラリ
import matplotlib.animation as animation

#%%

## 近似事後分布の推移をgif画像化

# 画像サイズを指定
fig = plt.figure(figsize=(12, 9))

# 作図処理を関数として定義
def update_posterior(i):
    # i回目の同時近似事後分布を計算
    E_tau = trace_a_i[i] / trace_b_i[i] # tauの期待値
    posterior_dens = norm.pdf(
        x=mu_grid.flatten(), 
        loc=trace_mu_i[i], 
        scale=np.sqrt(1.0 / (trace_lambda_i[i] / E_tau * tau_grid.flatten() + 1e-7))
    )
    posterior_dens *= gamma.pdf(
        x=tau_grid.flatten(), a=trace_a_i[i], scale=1.0 / trace_b_i[i]
    )
    
    # 前フレームのグラフを初期化
    plt.cla()
    
    # 近似事後分布を作図
    plt.scatter(x=mu_truth, y=tau_truth, color='red', s=100, marker='x', label='true val') # 真の値
    plt.contour(mu_grid, tau_grid, posterior_truth_dens.reshape(point_dims), 
                alpha=0.5, linestyles='--') # 真の事後分布
    plt.contour(mu_grid, tau_grid, posterior_dens.reshape(point_dims)) # 近似事後分布
    plt.xlabel('$\mu$')
    plt.ylabel('$\\tau$')
    plt.suptitle('Gaussian-Gamma Distribution', fontsize=20)
    plt.title('$iter:' + str(i * 0.5) + ', N=' + str(N) + 
              ', \mu_N=' + str(np.round(trace_mu_i[i], 1))+ 
              ', \lambda_N=' + str(np.round(trace_lambda_i[i], 5)) + 
              ', a_N=' + str(trace_a_i[i]) + 
              ', b_N=' + str(np.round(trace_b_i[i], 1)) + '$', 
              loc='left')
    plt.xlim(mu_truth - np.sqrt(1.0 / tau_truth), mu_truth + np.sqrt(1.0 / tau_truth))
    plt.ylim(0.0, 2.0 * tau_truth)
    plt.legend() # 凡例
    plt.grid() # グリッド線

# gif画像を作成
posterior_anime = animation.FuncAnimation(fig, update_posterior, frames=MaxIter * 2 + 1, interval=200)
posterior_anime.save("ch10_1_3_Posterior.gif")


#%%

print('end')

