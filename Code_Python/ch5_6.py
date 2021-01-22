# ch5.6 混合密度ネットワーク

#%%

# 利用するライブラリ
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#%%

## データを生成

# データ生成関数を定義
def fn(x):
    t = [n + 0.3 * np.sin(2 * np.pi * n) for n in x]
    eps = np.random.uniform(-0.1, 0.1, len(x))
    return t + eps

# データ数を指定
N = 300

# データを生成
t = np.random.uniform(0.0, 1.0, N)
x = fn(t)

# 作図
plt.scatter(x, t)
plt.xlabel('t')
plt.ylabel('x')
plt.title('$x_n = t_n + 0.3 \sin(2 \pi t_n) + \epsilon_n$', fontsize=20)
plt.show()

#%%

## 混合密度ネットワークの構築

'''
1次元ガウス分布にのみ対応
'''

# 混合密度ネットワークの定義
class MixtureDensityNetwork:
    def __init__(self, hidden_size, cluster_size):
        # 形状を保存
        L = 1 # 入力層
        H = hidden_size  # 中間層
        K = cluster_size # クラスタ数
        
        # ネットワークのパラメータを生成
        W1 = np.random.randn(L, H) * 0.01
        b1 = np.zeros(H)
        W2 = np.random.randn(H, 3 * K) * 0.01
        b2 = np.zeros(3 * K)
        self.network_params = [W1, b1, W2, b2] # 格納
    
    # モデルのパラメータ推定メソッド
    def predict(self, x):
        # ネットワークのパラメータを取得
        W1, b1, W2, b2 = self.network_params
        K = int(len(b2) / 3)
        
        # 全結合層の計算
        z = np.tanh(np.dot(x, W1) + b1) # 入力層
        a = np.dot(z, W2) + b2 # 出力層
        
        # モデルのパラメータを計算
        pi = np.exp(a[:, :K] - np.max(a[:, :K], axis=1, keepdims=True)) # オーバーフロー対策
        pi /= np.sum(pi, axis=1, keepdims=True) # 混合比率:式(5.150)
        sigma = np.exp(a[:, K:2*K]) # 標準偏差:式(5.151)
        mu = a[:, 2*K:] # 平均:式(5.152)
        
        # 逆伝播用に変数を保存
        self.z = z
        
        return pi, mu, sigma
    
    # 順伝播メソッド
    def forward(self, x, t):
        # モデルのパラメータを推定
        pi, mu, sigma = self.predict(x)
        
        # 誤差を計算:式(5.153)
        gaussian_density = np.exp(-0.5 * (mu - t) ** 2 / np.square(sigma)) / np.sqrt(2 * np.pi * np.square(sigma))
        #gaussian_density = norm.pdf(t, mu, sigma)
        self.gamma = pi * gaussian_density
        E = - np.sum(np.log(np.sum(self.gamma, axis=1)) + 1e-7)
        
        # 逆伝播用に計算
        self.gamma /= np.sum(self.gamma + 1e-7, axis=1, keepdims=True)
        
        # 逆伝播用に変数を保存
        self.ovservation_data = [x, t]
        self.model_params = [pi, mu, sigma]
        
        return E
    
    # 逆伝播メソッド
    def backward(self):
        # 変数を取得
        x, t = self.ovservation_data
        W1, b1, W2, b2 = self.network_params
        pi, mu, sigma = self.model_params
        L = 1 # 入力の次元：len(x[0])
        
        # モデルのパラメータの勾配を計算
        da_pi = pi - self.gamma # 混合比率:式(5.155)
        da_sigma = self.gamma * (L - np.sum((t - mu)**2, axis=1, keepdims=True) / sigma**2) # 分散:式(5.157)
        da_mu = self.gamma * (mu - t) / sigma**2 # 平均:式(5.156)
        da = np.hstack((da_pi, da_mu, da_sigma))
        
        # ネットワークの変数の勾配を計算
        dz = np.dot(da, W2.T)
        dW2 = np.dot(self.z.T, da)
        db2 = np.sum(da, axis=0)
        dx = np.dot(dz, W1.T)
        dW1 = np.dot(x.T, dz)
        db1 = np.sum(dz, axis=0)
        
        self.network_grads = [dW1, db1, dW2, db2] # 格納
    
    # パラメータの更新メソッド
    def update(self, lr=0.1):
        # SGDによる更新
        for i in range(4): # ネットワークのパラメータ数
            self.network_params[i] -= lr * self.network_grads[i]

#%%

# バッチサイズを指定
N = 30

# エポック当たりの試行回数を指定
max_epoch = 1000
max_iters = max(int(len(x) / N), 1)
print(max_iters)

#%%

## 推論

# 学習率を指定
lr = 0.00001

# 混合密度ネットワークのインスタンスを作成
network = MixtureDensityNetwork(hidden_size=5, cluster_size=3)

# 推定
for epoch in range(max_epoch):
    for iter in range(max_iters):
        # ミニバッチを取得
        idx = np.random.choice(len(x), N, replace=False)
        batch_x = x[idx].reshape(N, 1)
        batch_t = t[idx].reshape(N, 1)
        
        # 損失を計算
        E = network.forward(batch_x, batch_t)
        
        # 勾配を計算
        network.backward()
        
        # パラメータを更新
        network.update(lr)
    
    # 学習率を更新
    if (epoch + 1) % 1000 == 0:
        lr *= 0.9
    
    # 途中経過を表示
    print('epoch: ' + str(epoch + 1) + ', E=' + str(E))

print(lr)

#%%

## 結果の確認

# 作図用の格子状の点(データ)を作成
test_x = np.arange(0.0, 1.0, 0.01)
test_t = np.arange(0.0, 1.0, 0.01)
#test_x = np.linspace(min(x), max(x), 100)
#test_t = np.linspace(min(t), max(t), 100)
test_X, test_T = np.meshgrid(test_x, test_t)

# 各点(データ)の確率を計算
pi, mu, sigma = network.predict(test_X.reshape(-1, 1))
print(mu)
probs = np.exp(-0.5 * (mu - test_T.reshape(-1, 1)) ** 2 / np.square(sigma))
probs /= np.sqrt(2 * np.pi * np.square(sigma))
probs = norm.pdf(test_T.reshape(-1, 1), mu, sigma)
print(np.round(probs, 2))
probs = np.sum(pi * probs, axis=1)
Probs = probs.reshape(len(test_x), len(test_t))

# 作図
plt.scatter(x, t, alpha=0.5, label="observation")
levels_log = np.linspace(0, np.log(probs.max()), 5)
levels = np.exp(levels_log)
levels[0] = 0
plt.contourf(
    test_X, test_T, Probs, 
#    levels.reshape(len(test_x), len(test_t)), 
    alpha=0.5
)
#plt.colorbar()
plt.xlim(min(test_x), max(test_x))
plt.ylim(min(test_t), max(test_t))
plt.title('Mixture Density Network', fontsize=20)
plt.xlabel('x')
plt.ylabel('t')
plt.show()

#%%

# クラス内の処理の確認

#%%

## ガウス分布の確率密度の計算について

#　パラメータを推定
pi, mu, sigma = network.predict(test_X.reshape(-1, 1))

# 式通り計算
dens1 = np.exp(-0.5 * (mu - test_T.reshape(-1, 1)) ** 2 / sigma**2)
dens1 /= np.sqrt(2 * np.pi * sigma**2)
print(np.round(dens1, 5))

# scipyを利用
dens2 = norm.pdf(test_T.reshape(-1, 1), mu, sigma)
print(np.round(dens2, 5))

#%%

## 変数を作成

# 形状を保存
L = 1 # 入力層
H = 5 # 中間層
K = 3 # クラスタ数

# 観測データを取得
x = batch_x
t = batch_t

# ネットワークのパラメータを生成
W1 = np.random.randn(L, H) * 0.01
b1 = np.zeros(H)
W2 = np.random.randn(H, 3 * K) * 0.01
b2 = np.zeros(3 * K)

# ネットワークのパラメータを取得
W1, b1, W2, b2 = network.network_params
print(np.round(W1, 3))
print(np.round(b1, 3))
print(np.round(W2, 3))
print(np.round(b2, 3))

#%%

## 順伝播の計算

# 全結合層の計算
z = np.tanh(np.dot(x, W1) + b1) # 入力層
a = np.dot(z, W2) + b2 # 出力層

# 混合比率を計算:式(5.150)
pi = np.exp(a[:, :K] - np.max(a[:, :K], axis=1, keepdims=True)) # オーバーフロー対策
pi /= np.sum(pi, axis=1, keepdims=True)
print(np.round(pi[0:5], 3))
print(pi.shape)

# 標準偏差を計算:式(5.151)
sigma = np.exp(a[:, K:2*K])
print(np.round(sigma[0:5], 3))
print(sigma.shape)

# 平均を計算:式(5.152)
mu = a[:, 2*K:]
print(np.round(mu[0:5], 3))
print(mu.shape)

#%%

# 誤差を計算:式(5.153)
#gaussian_density = np.exp(-0.5 * (mu - t) ** 2 / np.square(sigma)) / np.sqrt(2 * np.pi * np.square(sigma))
gaussian_density = norm.pdf(t, mu, sigma)
gamma = pi * gaussian_density
print(np.round(pi, 3))
print(np.round(gaussian_density, 3))
E = - np.sum(np.log(np.sum(gamma, axis=1)))
print(E)

#%%

## 逆伝播の計算
gamma /= np.sum(gamma + 1e-7, axis=1, keepdims=True)

# モデルのパラメータの勾配を計算
da_pi = pi - gamma # 混合比率:式(5.155)
da_sigma = gamma * (L - (t - mu)**2 / sigma**2) # 分散:式(5.157)
da_mu = gamma * (mu - t) / sigma**2 # 平均:式(5.156)
da = np.hstack((da_pi, da_mu, da_sigma))
print(np.round(da, 3))
print(da.shape)

# ネットワークの変数の勾配を計算
dz = np.dot(da, W2.T)
dW2 = np.dot(z.T, da)
db2 = np.sum(da, axis=0)
dx = np.dot(dz, W1.T)
dW1 = np.dot(x.T, dz)
db1 = np.sum(dz, axis=0)
print(np.round(dW1, 3))
print(np.round(db1, 3))
print(np.round(dW2, 3))
print(np.round(db2, 3))

#%%
print(probs)
