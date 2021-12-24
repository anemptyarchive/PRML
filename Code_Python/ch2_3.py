### ch2.3.1-2 条件付きガウス分布と周辺ガウス分布

#%%

# 2.3.1-2項で利用するライブラリ
import numpy as np
from scipy.stats import norm, multivariate_normal # 1変量ガウス分布, 多変量ガウス分布
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#%%

## 同時ガウス分布の設定

# インデックスを設定:(固定)
a = 0
b = 1

# 平均パラメータを指定
mu_d = np.array([0.0, 0.0])

# 分散共分散行列を指定
sigma_dd = np.array([[20.0, -10.0], [-10.0, 7.0]])

# 精度行列を計算
lambda_dd = np.linalg.inv(sigma_dd)

# 作図用のxのx軸の値を作成
x_a = np.linspace(
    mu_d[a] - 3 * np.sqrt(sigma_dd[a, a]), 
    mu_d[a] + 3 * np.sqrt(sigma_dd[a, a]), 
    num=500
)

# 作図用のxのy軸の値を作成
x_b = np.linspace(
    mu_d[b] - 3 * np.sqrt(sigma_dd[b, b]), 
    mu_d[b] + 3 * np.sqrt(sigma_dd[b, b]), 
    num=500
)

# 作図用のxの点を作成
X_a, X_b = np.meshgrid(x_a, x_b)
x_point = np.stack([X_a.flatten(), X_b.flatten()], axis=1)
x_dims = X_a.shape

# 同時ガウス分布を計算
joint_dens = multivariate_normal.pdf(
    x=x_point, mean=mu_d, cov=sigma_dd
)

#%%

# 同時ガウス分布の2Dグラフを作成
plt.figure(figsize=(9, 8))
plt.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet') # 同時分布:(等高線図)
plt.xlabel('$x_a$')
plt.ylabel('$x_b$')
plt.suptitle('Joint Gaussian Distribution', fontsize=20)
plt.title('$\mu=(' + ', '.join([str(mu) for mu in mu_d]) + ')' + 
          ', \Sigma=' + str([list(sgm_d) for sgm_d in np.round(sigma_dd, 1)]) + '$', loc='left')
plt.grid()
plt.colorbar(label='density')
plt.gca().set_aspect('equal')
plt.show()

#%%

# 同時ガウス分布の3Dグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet') # 同時分布:(曲面図)
#ax.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', offset=0) # 同時分布:(等高線図)
ax.set_xlabel('$x_a$')
ax.set_ylabel('$x_b$')
ax.set_zlabel('density')
plt.title('$\mu=(' + ', '.join([str(mu) for mu in mu_d]) + ')' + 
          ', \Sigma=' + str([list(sgm_d) for sgm_d in np.round(sigma_dd, 1)]) + '$', 
          loc='left')
fig.suptitle('Joint Gaussian Distribution', fontsize=20)
#ax.view_init(elev=0, azim=300) # 表示アングル
plt.show()


#%%

## 条件付き分布の計算

# xの点を指定
x_d = np.array([-7.0, 3.0])

# x1の条件付きガウス分布のパラメータを計算
mu_a = mu_d[a] + sigma_dd[a, b] / sigma_dd[b, b] * (x_d[b] - mu_d[b])
sigma_a = sigma_dd[a, a] - sigma_dd[a, b] / sigma_dd[b, b] * sigma_dd[b, a]
print(mu_a)
print(sigma_a)
mu_a = mu_d[a] - (1.0 / lambda_dd[a, a]) * lambda_dd[a, b] * (x_d[b] - mu_d[b])
sigma_a = 1.0 / lambda_dd[a, a]
print(mu_a)
print(sigma_a)

# x2の条件付きガウス分布のパラメータを計算
mu_b = mu_d[b] + sigma_dd[b, a] / sigma_dd[a, a] * (x_d[a] - mu_d[a])
sigma_b = sigma_dd[b, b] - sigma_dd[b, a] / sigma_dd[a, a] * sigma_dd[a, b]
print(mu_b)
print(sigma_b)
mu_b = mu_d[b] - (1.0 / lambda_dd[b, b]) * lambda_dd[b, a] * (x_d[a] - mu_d[a])
sigma_b = 1.0 / lambda_dd[b, b]
print(mu_b)
print(sigma_b)

# 条件付きガウス分布を計算
conditional_dens_a = norm.pdf(x=x_a, loc=mu_a, scale=np.sqrt(sigma_a))
conditional_dens_b = norm.pdf(x=x_b, loc=mu_b, scale=np.sqrt(sigma_b))

#%%

# 条件付きガウス分布の3Dグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', alpha=0.5) # 同時分布:(曲面図)
ax.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', linestyles='--', offset=0) # 同時分布:(等高線図)
ax.plot(x_a, np.repeat(x_d[b], len(x_a)), conditional_dens_a, color='darkturquoise', label='$p(x_a | x_b)$') # x1の条件付き分布
ax.plot(x_a, np.repeat(x_d[b], len(x_a)), np.repeat(0, len(x_a)), color='darkturquoise', linestyle=':') # x1の条件付き分布の補助線
ax.plot(np.repeat(x_d[a], len(x_b)), x_b, conditional_dens_b, color='blue', label='$p(x_b | x_a)$') # x2の条件付き分布
ax.plot(np.repeat(x_d[a], len(x_b)), x_b, np.repeat(0, len(x_b)), color='blue', linestyle=':') # x2の条件付き分布の補助線
ax.scatter(x_d[a], x_d[b], color='orange', marker='+', s=200, label='$(x_a, x_b)$') # xの点
ax.set_xlabel('$x_a$')
ax.set_ylabel('$x_b$')
ax.set_zlabel('density')
ax.set_title('$x=(' + ', '.join([str(x) for x in np.round(x_d, 1)]) + ')' + 
             ', \mu_{a|b}=' + str(np.round(mu_a, 1)) + ', \Sigma_{a|b}=' + str(np.round(sigma_a, 1)) + 
             ', \mu_{b|a}=' + str(np.round(mu_b, 1)) + ', \Sigma_{b|a}=' + str(np.round(sigma_b, 1)) + '$', loc='left')
fig.suptitle('Conditional Gaussian Distribution', fontsize=20)
ax.legend()
#ax.view_init(elev=0, azim=300) # 表示アングル
plt.show()

#%%

# 1つの変数を固定した同時ガウス分布の確率密度を計算
joint_dens_a = multivariate_normal.pdf(
    x=np.stack([x_a, np.repeat(x_d[b], len(x_a))], axis=1), mean=mu_d, cov=sigma_dd
)
joint_dens_b = multivariate_normal.pdf(
    x=np.stack([np.repeat(x_d[a], len(x_b)), x_b], axis=1), mean=mu_d, cov=sigma_dd
)

#%%

# 条件付きガウス分布の3Dグラフを作成
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
#ax.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', linestyles=':', offset=0) # 同時分布:(等高線図)
ax.plot(x_a, np.repeat(x_d[b], len(x_a)), joint_dens_a, color='darkturquoise', linestyle='--', label='$p(x_a, x_b=' + str(x_d[b]) + ')$') # x2を固定した同時分布
ax.plot(x_a, np.repeat(x_d[b], len(x_a)), conditional_dens_a, color='darkturquoise', label='$p(x_a | x_b=' + str(x_d[b]) + ')$') # x1の条件付き分布
ax.plot(np.repeat(x_d[a], len(x_b)), x_b, joint_dens_b, color='blue', linestyle='--', label='$p(x_a=' + str(x_d[a]) + ', x_b)$') # x1を固定した同時分布
ax.plot(np.repeat(x_d[a], len(x_b)), x_b, conditional_dens_b, color='blue', label='$p(x_b | x_a=' + str(x_d[a]) + ')$') # x2の条件付き分布
ax.scatter(x_d[a], x_d[b], color='orange', marker='+', s=200, label='$(x_a=' + str(x_d[a]) + ', x_b=' + str(x_d[b]) + ')$') # xの点
ax.set_xlabel('$x_a$')
ax.set_ylabel('$x_b$')
ax.set_zlabel('density')
ax.set_title('$\mu_{a|b}=' + str(np.round(mu_a, 1)) + ', \Sigma_{a|b}=' + str(np.round(sigma_a, 1)) + 
             ', \mu_{b|a}=' + str(np.round(mu_b, 1)) + ', \Sigma_{b|a}=' + str(np.round(sigma_b, 1)) + '$', loc='left')
fig.suptitle('Conditional Gaussian Distribution', fontsize=20)
ax.legend(bbox_to_anchor=(0.9, 0.8), loc='upper left')
ax.view_init(elev=0, azim=270) # 表示アングル
plt.show()

#%%

# フレーム数を指定
n_frame = 100

# 使用するpの値を作成
x_a_vals = np.linspace(np.min(x_a), np.max(x_a), num=n_frame) # x軸の値
x_b_vals = np.linspace(np.min(x_b), np.max(x_b), num=n_frame) # y軸の値

# 図を初期化
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Conditional Gaussian Distribution', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のxの値を取得
    x_d[a] = x_a_vals[i]
    x_d[b] = x_b_vals[i]
    
    # x1の条件付きガウス分布のパラメータを計算
    mu_a = mu_d[a] + sigma_dd[a, b] / sigma_dd[b, b] * (x_d[b] - mu_d[b])
    sigma_a = sigma_dd[a, a] - sigma_dd[a, b] / sigma_dd[b, b] * sigma_dd[b, a]
    
    # x2の条件付きガウス分布のパラメータを計算
    mu_b = mu_d[b] + sigma_dd[b, a] / sigma_dd[a, a] * (x_d[a] - mu_d[a])
    sigma_b = sigma_dd[b, b] - sigma_dd[b, a] / sigma_dd[a, a] * sigma_dd[a, b]
    
    # 条件付きガウス分布を計算
    conditional_dens_a = norm.pdf(x=x_a, loc=mu_a, scale=np.sqrt(sigma_a))
    conditional_dens_b = norm.pdf(x=x_b, loc=mu_b, scale=np.sqrt(sigma_b))
    
    # 条件付きガウス分布の3Dグラフを作成
    #ax.plot_surface(X_a, X_b, multivariate_dens.reshape(x_dims), cmap='jet', alpha=0.5) # 同時分布:(曲面図)
    ax.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', linestyles='--', offset=0) # 同時分布:(等高線図)
    ax.plot(x_a, np.repeat(x_d[b], len(x_a)), conditional_dens_a, color='darkturquoise', label='$p(x_a | x_b)$') # x1の条件付き分布
    ax.plot(x_a, np.repeat(x_d[b], len(x_a)), np.repeat(0, len(x_a)), color='darkturquoise', linestyle=':') # x1の条件付き分布の補助線
    ax.plot(np.repeat(x_d[a], len(x_b)), x_b, conditional_dens_b, color='blue', label='$p(x_b | x_a)$') # x2の条件付き分布
    ax.plot(np.repeat(x_d[a], len(x_b)), x_b, np.repeat(0, len(x_b)), color='blue', linestyle=':') # x2の条件付き分布の補助線
    ax.scatter(x_d[a], x_d[b], color='orange', marker='+', s=200, label='$(x_a, x_b)$') # xの点
    ax.set_xlabel('$x_a$')
    ax.set_ylabel('$x_b$')
    ax.set_zlabel('density')
    ax.set_title('$x=(' + ', '.join([str(x) for x in np.round(x_d, 1)]) + ')' + 
                 ', \mu_{a|b}=' + str(np.round(mu_a, 1)) + ', \Sigma_{a|b}=' + str(np.round(sigma_a, 1)) + 
                 ', \mu_{b|a}=' + str(np.round(mu_b, 1)) + ', \Sigma_{b|a}=' + str(np.round(sigma_b, 1)) + '$', loc='left')
    ax.legend()
    #ax.view_init(elev=0, azim=270) # 表示アングル

# gif画像を作成
anime_conditional = FuncAnimation(fig, update, frames=n_frame, interval=100)

# gif画像を保存
anime_conditional.save('PRML/Fig/ch2_3_1_ConditionalDistribution_1.gif')

#%%

# 図を初期化
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Conditional Gaussian Distribution', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目のxの値を取得
    x_d[a] = x_a_vals[i]
    x_d[b] = x_b_vals[i]
    
    # x1の条件付きガウス分布のパラメータを計算
    mu_a = mu_d[a] + sigma_dd[a, b] / sigma_dd[b, b] * (x_d[b] - mu_d[b])
    sigma_a = sigma_dd[a, a] - sigma_dd[a, b] / sigma_dd[b, b] * sigma_dd[b, a]
    
    # x2の条件付きガウス分布のパラメータを計算
    mu_b = mu_d[b] + sigma_dd[b, a] / sigma_dd[a, a] * (x_d[a] - mu_d[a])
    sigma_b = sigma_dd[b, b] - sigma_dd[b, a] / sigma_dd[a, a] * sigma_dd[a, b]
    
    # 条件付きガウス分布を計算
    conditional_dens_a = norm.pdf(x=x_a, loc=mu_a, scale=np.sqrt(sigma_a))
    conditional_dens_b = norm.pdf(x=x_b, loc=mu_b, scale=np.sqrt(sigma_b))
    
    # 1つの変数を固定した同時ガウス分布の確率密度を計算
    joint_dens_a = multivariate_normal.pdf(
        x=np.stack([x_a, np.repeat(x_d[b], len(x_a))], axis=1), mean=mu_d, cov=sigma_dd
    )
    joint_dens_b = multivariate_normal.pdf(
        x=np.stack([np.repeat(x_d[a], len(x_b)), x_b], axis=1), mean=mu_d, cov=sigma_dd
    )
    
    # 条件付きガウス分布の3Dグラフを作成
    ax.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', linestyles=':', offset=0) # 同時分布:(等高線図)
    ax.plot(x_a, np.repeat(x_d[b], len(x_a)), joint_dens_a, color='darkturquoise', linestyle='--', label='$p(x_a, x_b=' + str(np.round(x_d[b], 1)) + ')$') # x2を固定した同時分布
    ax.plot(x_a, np.repeat(x_d[b], len(x_a)), conditional_dens_a, color='darkturquoise', label='$p(x_a | x_b=' + str(np.round(x_d[b], 1)) + ')$') # x1の条件付き分布
    ax.plot(np.repeat(x_d[a], len(x_b)), x_b, joint_dens_b, color='blue', linestyle='--', label='$p(x_a=' + str(np.round(x_d[a], 1)) + ', x_b)$') # x1を固定した同時分布
    ax.plot(np.repeat(x_d[a], len(x_b)), x_b, conditional_dens_b, color='blue', label='$p(x_b | x_a=' + str(np.round(x_d[a], 1)) + ')$') # x2の条件付き分布
    ax.scatter(x_d[a], x_d[b], color='orange', marker='+', s=200, label='$(x_a=' + str(np.round(x_d[a], 1)) + ', x_b=' + str(np.round(x_d[b], 1)) + ')$') # xの点
    ax.set_xlabel('$x_a$')
    ax.set_ylabel('$x_b$')
    ax.set_zlabel('density')
    ax.set_title('$\mu_{a|b}=' + str(np.round(mu_a, 1)) + ', \Sigma_{a|b}=' + str(np.round(sigma_a, 1)) + 
                 ', \mu_{b|a}=' + str(np.round(mu_b, 1)) + ', \Sigma_{b|a}=' + str(np.round(sigma_b, 1)) + '$', loc='left')
    ax.legend(bbox_to_anchor=(0.9, 1.1), loc='upper left')
    #ax.view_init(elev=0, azim=270) # 表示アングル

# gif画像を作成
anime_conditional = FuncAnimation(fig, update, frames=n_frame, interval=100)

# gif画像を保存
anime_conditional.save('PRML/Fig/ch2_3_1_ConditionalDistribution_2.gif')


#%%

## 周辺ガウス分布の計算

# 周辺ガウス分布を計算
marginal_dens_a = norm.pdf(x=x_a, loc=mu_d[a], scale=np.sqrt(sigma_dd[a, a]))
marginal_dens_b = norm.pdf(x=x_b, loc=mu_d[b], scale=np.sqrt(sigma_dd[b, b]))
print(sigma_dd[a, a])
print(1.0 / (lambda_dd[a, a] - lambda_dd[a, b] / lambda_dd[b, b] * lambda_dd[b, a]))
print(sigma_dd[b, b])
print(1.0 / (lambda_dd[b, b] - lambda_dd[b, a] / lambda_dd[a, a] * lambda_dd[a, b]))

#%%

# 周辺ガウス分布の3Dグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', alpha=0.5) # 同時分布:(曲面図)
#ax.contour(X_a, X_b, joint_dens.reshape(x_dims), cmap='jet', linestyles='--', offset=0) # 同時分布:(等高線図)
ax.plot(x_a, np.repeat(np.max(x_b), len(x_a)), marginal_dens_a, color='darkturquoise', label='$p(x_a)$') # x1の周辺分布
ax.plot(x_a, np.repeat(np.max(x_b), len(x_a)), np.repeat(0, len(x_a)), color='darkturquoise', linestyle=':') # x1の周辺分布の補助線
ax.plot(np.repeat(np.max(x_a), len(x_b)), x_b, marginal_dens_b, color='blue', label='$p(x_b)$') # x2の周辺分布
ax.plot(np.repeat(np.max(x_a), len(x_b)), x_b, np.repeat(0, len(x_b)), color='blue', linestyle=':') # x2の周辺分布の補助線
ax.set_xlabel('$x_a$')
ax.set_ylabel('$x_b$')
ax.set_zlabel('density')
ax.set_title('$\mu_a=' + str(np.round(mu_d[a], 1)) + ', \Sigma_{a,a}=' + str(np.round(sigma_dd[a, a], 1)) + 
             ', \mu_b=' + str(np.round(mu_d[b], 1)) + ', \Sigma_{b,b}=' + str(np.round(sigma_dd[b, b], 1)) + '$', loc='left')
fig.suptitle('Marginal Gaussian Distribution', fontsize=20)
ax.legend()
#ax.view_init(elev=0, azim=270) # 表示アングル
plt.show()


#%%

print('end')

