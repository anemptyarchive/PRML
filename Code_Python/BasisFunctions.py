# 基底関数

# 利用するライブラリ
import numpy as np

#%%

### 利用する関数

# ロジスティックシグモイド関数を作成
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

#%%

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
def Phi_polynomial(x_n, M, _x_n=None):
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
    
    # M-1個のパラメータを作成
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
    
    # M-1個のパラメータを作成
    s = np.std(_x_n) * 0.5 # 標準偏差で固定(0.5は値の調整)
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

#%%

### 2次元の場合

## 基底関数

# 2次元多項式基底関数を作成
def phi_polynomial2d(x_d, j):
    # データごとに総和を計算
    return np.sum(x_d**j, axis=1)

# 2次元ガウス基底関数を作成
def phi_gauss2d(x_d, mu_d, s_d):
        # m番目のパラメータを使って二次形式の計算
        a_d = -0.5 * (x_d - mu_d)**2 / s_d**2
        
        # データごとに総和を計算
        a = np.sum(a_d, axis=1)
        
        # 指数をとる
        return np.exp(a)

# 2次元シグモイド基底関数を作成
def phi_sigmoid2d(x_d, mu_d, s_d):
        # m番目のパラメータにより入力値を調整
        a_d = (x_d - mu_d) / s_d
        
        # ロジスティックシグモイド関数の計算
        y_d = sigmoid(a_d)
        
        # データごとに総和を計算
        return np.sum(y_d, axis=1) / len(s_d)


## 計画行列

# 2次元多項式基底関数の計画行列を作成
def Phi_polynomial2d(x_nd, M, _x_nd=None):
    # 変数を初期化
    phi_x_nm = np.ones((len(x_nd), M))
    
    # 2次元多項式基底関数による変換
    for m in range(1, M):
        phi_x_nm[:, m] = phi_polynomial2d(x_nd, m)
    return phi_x_nm

# 2次元ガウス基底関数の計画行列を作成:(対角線に標準化)
def Phi_gauss2d(x_nd, M, _x_nd=None):
    # パラメータ設定用の入力値を設定
    if _x_nd is None:
        _x_nd = x_nd.copy()
    
    # M-1個のパラメータを作成
    s_d = np.std(_x_nd, axis=0) # 標準偏差で固定
    if M == 2:
        # 入力値の平均
        mu_md = np.array([[np.mean(_x_nd[:, 0]), np.mean(_x_nd[:, 1])]])
    elif M == 3:
        # 調整幅を指定
        sgm_d = s_d * 2.0
        
        # 入力値の平均
        mu_md = np.array(
            [[np.mean(_x_nd[:, 0]) - sgm_d[0], np.mean(_x_nd[:, 0]) - sgm_d[1]], 
             [np.mean(_x_nd[:, 0]) + sgm_d[0], np.mean(_x_nd[:, 0]) + sgm_d[1]]]
        )
    elif M > 3:
        # 調整幅を指定
        sgm_d = s_d * 0.5
        
        # 最小値から最大値を等分
        mu_md = np.stack([
            np.linspace(np.min(_x_nd[:, 0]) + sgm_d[0], np.max(_x_nd[:, 0] - sgm_d[1]), num=M-1), 
            np.linspace(np.min(_x_nd[:, 1]) + sgm_d[0], np.max(_x_nd[:, 1] - sgm_d[1]), num=M-1)
        ], axis=1)
    
    # 変数を初期化
    phi_x_nm = np.ones((len(x_nd), M))
    
    # 2次元ガウス基底関数による変換
    for m in range(1, M):
        phi_x_nm[:, m] = phi_gauss2d(x_nd, mu_md[m-1], s_d)
    return phi_x_nm

# 2次元シグモイド基底関数の計画行列を作成:(対角線に標準化)
def Phi_sigmoid2d(x_nd, M, _x_nd=None):
    # パラメータ設定用の入力値を設定
    if _x_nd is None:
        _x_nd = x_nd.copy()
    
    # M-1個のパラメータを作成
    s_d = np.std(_x_nd, axis=0) # 標準偏差で固定
    if M == 2:
        # 入力値の平均
        mu_md = np.array([[np.mean(_x_nd[:, 0]), np.mean(_x_nd[:, 1])]])
    elif M == 3:
        # 調整幅を指定
        sgm_d = s_d * 2.0
        
        # 入力値の平均
        mu_md = np.array(
            [[np.mean(_x_nd[:, 0]) - sgm_d[0], np.mean(_x_nd[:, 0]) - sgm_d[1]], 
             [np.mean(_x_nd[:, 0]) + sgm_d[0], np.mean(_x_nd[:, 0]) + sgm_d[1]]]
        )
    elif M > 3:
        # 調整幅を指定
        sgm_d = s_d * 0.5
        
        # 最小値から最大値を等分
        mu_md = np.stack([
            np.linspace(np.min(_x_nd[:, 0]) + sgm_d[0], np.max(_x_nd[:, 0] - sgm_d[1]), num=M-1), 
            np.linspace(np.min(_x_nd[:, 1]) + sgm_d[0], np.max(_x_nd[:, 1] - sgm_d[1]), num=M-1)
        ], axis=1)
    
    # 変数を初期化
    phi_x_nm = np.ones((len(x_nd), M))
    
    # 2次元シグモイド基底関数による変換
    for m in range(1, M):
        phi_x_nm[:, m] = phi_sigmoid2d(x_nd, mu_md[m-1], s_d)
    return phi_x_nm

