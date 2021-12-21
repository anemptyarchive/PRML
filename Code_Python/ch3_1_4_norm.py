### ch3.1.4 Lpノルムの作図

#%%

# 3.1.4項で利用するライブラリ
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#%%

## Lpノルムの作図

# 値を指定
p = 1

# wの値を指定
w_vals = np.arange(-10.0, 10.1, 0.1)

# 作図用のwの点を作成
W1, W2 = np.meshgrid(w_vals, w_vals)

# Lpのノルムを計算
Lp = (np.abs(W1)**p + np.abs(W2)**p)**(1.0 / p)

#%%

# Lpノルムの3Dグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(W1, W2, Lp, cmap='jet') # 曲面図
ax.contour(W1, W2, Lp, cmap='jet', offset=0) # 等高線図
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('$||w||_p$')
ax.set_title('p=' + str(np.round(p, 1)), loc='left')
fig.suptitle('$||w||_p = {}^p\sqrt{\sum_{j=1}^M |w_j|^p}$')
#ax.view_init(elev=90, azim=270) # 表示アングル
plt.show()

#%%

# Lpノルムの2Dグラフを作成
plt.figure(figsize=(9, 8))
plt.contour(W1, W2, Lp, cmap='jet') # 等高線図
#plt.contour(W1, W2, Lp, cmap='jet', levels=1) # 等高線図:(値を指定)
#plt.contourf(W1, W2, Lp, cmap='jet') # 塗りつぶし等高線図
plt.xlabel('$w_1$')
plt.ylabel('$w_2$')
plt.title('p=' + str(np.round(p, 1)), loc='left')
plt.suptitle('$||w||_p = {}^p\sqrt{\sum_{j=1}^M |w_j|^p}$')
plt.colorbar(label='$||w||_p$')
plt.grid()
plt.gca().set_aspect('equal')
plt.show()

#%%

## 正則化項の作図

# 正則化項を計算
E_W = (np.abs(W1)**p + np.abs(W2)**p) / p

# 正則化項の3Dグラフを作成
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d') # 3D用の設定
ax.plot_surface(W1, W2, E_W, cmap='jet') # 曲面図
ax.contour(W1, W2, E_W, cmap='jet', offset=0) # 等高線図
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('$E_W(w)$')
ax.set_title('p=' + str(np.round(p, 1)), loc='left')
fig.suptitle('$E_W(w) = \\frac{1}{p} \sum_{j=1}^M |w_j|^p$')
#ax.view_init(elev=90, azim=270) # 表示アングル
plt.show()

#%%

## pとグラフの形状の関係

# 使用するpの値を指定
p_vals = np.arange(0.1, 10.1, 0.1)

# 図を初期化
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(projection='3d') # 3D用の設定
fig.suptitle('Lp-Norm', fontsize=20)

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目の値を取得
    p = p_vals[i]
    
    # Lpノルムを計算
    Lp = (np.abs(W1)**p + np.abs(W2)**p)**(1.0 / p)
    
    # Lpノルムの3Dグラフを作成
    ax.plot_surface(W1, W2, Lp, cmap='jet') # 曲面図
    ax.contour(W1, W2, Lp, cmap='jet', offset=0) # 等高線図
    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    ax.set_zlabel('$||w||_p$')
    ax.set_title('p=' + str(np.round(p, 1)), loc='left')

# gif画像を作成
anime_norm3d = FuncAnimation(fig, update, frames=len(p_vals), interval=100)

# gif画像を保存
anime_norm3d.save('PRML/Fig/ch3_1_4_LpNorm_3d.gif')

#%%

# 図を初期化
fig = plt.figure(figsize=(6, 6))

# 作図処理を関数として定義
def update(i):
    # 前フレームのグラフを初期化
    plt.cla()
    
    # i回目の値を取得
    p = p_vals[i]
    
    # Lpノルムを計算
    Lp = (np.abs(W1)**p + np.abs(W2)**p)**(1.0 / p)
    
    # Lpノルムの2Dグラフを作成
    plt.contour(W1, W2, Lp, cmap='jet') # 等高線図
    #plt.contourf(W1, W2, Lp, cmap='jet') # 塗りつぶし等高線図
    plt.xlabel('$w_1$')
    plt.ylabel('$w_2$')
    plt.title('p=' + str(np.round(p, 1)), loc='left')
    plt.suptitle('Lp-Norm', fontsize=20)
    plt.grid()
    plt.axes().set_aspect('equal')

# gif画像を作成
anime_norm2d = FuncAnimation(fig, update, frames=len(p_vals), interval=100)

# gif画像を保存
anime_norm2d.save('PRML/Fig/ch3_1_4_LpNorm_2d.gif')


