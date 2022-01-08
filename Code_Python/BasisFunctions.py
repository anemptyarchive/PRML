# 基底関数

# 利用するライブラリ
import numpy as np

#%%

### 利用する関数

# ロジスティックシグモイド関数を作成
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


### 1次元の場合

## 基底関数

# 多項式基底関数を作成
def phi_polynomial(x, j):
    return x**j

# ガウス基底関数を作成
def phi_gauss(x, mu, s):
    a_n = -0.5 * (x - mu)**2 / s**2
    return np.exp(a_n)

# シグモイド基底関数を作成
def phi_sigmoid(x, mu, s):
    a_n = (x - mu) / s
    return sigmoid(a_n)


## 計画行列

# 多項式基底関数の計画行列を作成:(M > 2)
def Phi_polynomial(x_n, M):
    # 変数を初期化
    phi_x_nm = np.ones((len(x_n), M))
    
    # 列ごとに多項式基底関数による変換
    for m in range(1, M):
        phi_x_nm[:, m] = phi_polynomial(x_n, m)
    return phi_x_nm

# ガウス基底関数の計画行列を作成:(M > 2)
def Phi_gauss(x_n, M, _x_n=None):
    # パラメータ設定用の入力値を設定
    if _x_n is None:
        _x_n = x_n.copy()
    
    # M-1個のパラメータを作成:(M > 1)
    s = np.std(_x_n) # 標準偏差で固定
    if M == 2:
        # 入力値の平均
        mu_m = np.array([np.mean(_x_n)])
    elif M == 3:
        # 調整幅を指定
        sgm = s * 1.0
        
        # 平均を中心に標準偏差のn倍の範囲
        mu_m = np.array([np.mean(_x_n) - sgm, np.mean(_x_n) + sgm])
    elif M > 3:
        # 調整幅を指定
        sgm = s * 0.25
        
        # 最小値から最大値を等分
        mu_m = np.linspace(np.min(_x_n) + sgm, np.max(_x_n) - sgm, num=M-1)
    
    # 変数を初期化
    phi_x_nm = np.ones((len(x_n), M))
    
    # 列ごとにガウス基底関数による変換
    for m in range(1, M):
        phi_x_nm[:, m] = phi_gauss(x_n, mu_m[m-1], s)
    return phi_x_nm

# シグモイド基底関数の計画行列を作成:(M > 2)
def Phi_sigmoid(x_n, M, _x_n=None):
    # パラメータ設定用の入力値を設定
    if _x_n is None:
        _x_n = x_n.copy()
    
    # M-1個のパラメータを作成:(M > 1)
    s = np.std(_x_n) * 0.5 # 標準偏差で固定
    if M == 2:
        # 入力値の平均
        mu_m = np.array([np.mean(_x_n)])
    elif M == 3:
        # 調整幅を指定
        sgm = s * 1.0
        
        # 平均を中心に標準偏差のn倍の範囲
        mu_m = np.array([np.mean(_x_n) - sgm, np.mean(_x_n) + sgm])
    elif M > 3:
        # 調整幅を指定
        sgm = s * 0.25
        
        # 最小値から最大値を等分
        mu_m = np.linspace(np.min(_x_n) + sgm, np.max(_x_n) - sgm, num=M-1)
    
    # 変数を初期化
    phi_x_nm = np.ones((len(x_n), M))
    
    # 列ごとにシグモイド基底関数による変換
    for m in range(1, M):
        phi_x_nm[:, m] = phi_sigmoid(x_n, mu_m[m-1], s)
    return phi_x_nm


### 2次元の場合

## 基底関数



## 計画行列
