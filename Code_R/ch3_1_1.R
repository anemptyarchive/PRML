
# ch3.1.1：最尤推定と最小二乗法 --------------------------------------------------------

# 利用するパッケージ
library(tidyverse)

# ch3_1_4_ridge.R, ch3_1_4_lasso.Rでも利用


### モデルの設定 -----

# 真の関数を作成
y_true <- function(x) {
  # 計算式を指定
  y <- sin(2 * pi * x)
  return(y)
}


# データの範囲を指定
x_min <- 0
x_max <- 1

# 作図用のxの値を作成
x_vals <- seq(x_min, x_max, by = 0.01)


# 真の精度パラメータを指定
beta_true <- 3

# 真の標準偏差を計算
sigma_true <- sqrt(1 / beta_true)
sigma_true


# 真のモデルを計算
model_df <- tidyr::tibble(
  x = x_vals, # x軸の値
  y = y_true(x_vals), # y軸の値
  minus_sigma = y - 2 * sigma_true, # μ - 2σ
  plus_sigma = y + 2 * sigma_true # μ + 2σ
)

# 真のモデルを作図
ggplot() + 
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_ribbon(data = model_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "cyan4", alpha = 0.1, color = "cyan4", linetype = "dotted") + # 真のノイズ範囲
  labs(title = expression(t == sin(2 * pi * x)), 
       subtitle = paste0("beta=", beta_true), 
       x = "x", y = "t")


### 基底関数の設定 -----

# 多項式基底関数を作成
phi_poly <- function(x, m) {
  # m乗を計算
  y <- x^m
  return(y)
}

# 計画行列を作成
Phi_poly <- function(x_n, M) {
  # 変数を初期化
  x_nm <- matrix(0, nrow = length(x_n), ncol = M)
  
  # m列目を計算
  for(m in 0:(M-1)) {
    x_nm[, m+1] <- phi_poly(x_n, m)
  }
  return(x_nm)
}


# ガウス基底関数を作成
phi_gauss <- function(x, mu = 0, s = 0.2) {
  # ガウス関数の計算
  y <- exp(-(x - mu)^2 / (2 * s^2))
  return(y)
}

# 計画行列を作成
Phi_gauss <- function(x_n, M) {
  # M-1個のパラメータを指定
  mu_m <- seq(0, 1, length.out = M-1)
  s <- 0.1
  
  # 変数を初期化
  x_nm <- matrix(1, nrow = length(x_n), ncol = M)
  
  # (1行目を除く)m列目を計算
  for(m in 1:(M-1)) {
    x_nm[, m+1] <- phi_gauss(x_n, mu = mu_m[m], s = s)
  }
  return(x_nm)
}


# シグモイド基底関数を作成
phi_sigmoid <- function(x, mu = 0, s = 0.1) {
  # 入力を標準化
  a <- (x - mu) / s
  
  # ロジスティックシグモイド関数の計算
  y <- 1 / (1 + exp(-a))
  return(y)
}

# 計画行列を作成
Phi_sigmoid <- function(x_n, M) {
  # M-1個のパラメータを指定
  mu_m <- seq(0, 1, length.out = M-1)
  s <- 0.1
  
  # 変数を初期化
  x_nm <- matrix(1, nrow = length(x_n), ncol = M)
  
  # (1行目を除く)m列目を計算
  for(m in 1:(M-1)) {
    x_nm[, m+1] <- phi_sigmoid(x_n, mu = mu_m[m], s = s)
  }
  return(x_nm)
}


# 基底関数を設定
Phi <- Phi_poly
Phi <- Phi_gauss
Phi <- Phi_sigmoid


### データの生成 -----

# データ数を指定
N <- 100

# (観測)データを生成
x_n <- runif(n = N, min = x_min, max = x_max) # 入力
t_n <- y_true(x_n) + rnorm(n = N, mean = 0, sd = sigma_true) # 出力


# 観測データをデータフレームに格納
data_df <- tidyr::tibble(
  x_n = x_n, # 入力
  t_n = t_n # 出力
)

# 観測データの散布図を作成
ggplot() + 
  geom_point(data = data_df, aes(x = x_n, y = t_n)) + # 観測データ
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_ribbon(data = model_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "cyan4", alpha = 0.1, color = "cyan4", linetype = "dotted") + # 真のノイズ範囲
  labs(title = expression(t[n] == sin(2 * pi * x[n]) + epsilon[n]), 
       subtitle = paste0("N=", N, ", beta=", beta_true), 
       x = "x", y = "t")


### 最尤推定 -----

# 重みパラメータの次元数(基底関数の数)を指定
M <- 6


# 基底関数により入力を変換
phi_x_nm <- Phi(x_n, M)

# 重みパラメータの最尤推定量を計算
w_ml_m <- solve(t(phi_x_nm) %*% phi_x_nm) %*% t(phi_x_nm) %*% t_n %>% 
  as.vector()

# 分散パラメータの最尤推定量を計算
sigma2_ml <- sum((t_n - phi_x_nm %*% w_ml_m)^2) / N

# 精度パラメータの最尤推定量を計算
beta_ml <- 1 / sigma2_ml


# 推定したパラメータによるモデルを計算
ml_df <- tidyr::tibble(
  x = x_vals, # x軸の値
  t = Phi(x_vals, M) %*% w_ml_m %>% 
    as.vector(), # y軸の値
  minus_sigma = t - 2 * sqrt(sigma2_ml), # μ - 2σ
  plus_sigma = t + 2 * sqrt(sigma2_ml) # μ + 2σ
)

# 推定したパラメータによるモデルを作図
ggplot() + 
  geom_point(data = data_df, aes(x = x_n, y = t_n)) + # 観測データ
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_ribbon(data = model_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "cyan4", alpha = 0.1, color = "cyan4", linetype = "dotted") + # 真のノイズ範囲
  geom_line(data = ml_df, aes(x = x, y = t), color = "blue") + # 推定したモデル
  geom_ribbon(data = ml_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "blue", alpha = 0.1, color = "blue", linetype = "dotted") + # 推定したノイズ範囲
  #ylim(c(-5, 5)) + # y軸の表示範囲
  labs(title = "Linear Basis Function Model", 
       subtitle = paste0("N=", N, ", M=", M, ", beta=", round(beta_ml, 2), 
                         ", w=(", paste0(round(w_ml_m, 2), collapse = ", "), ")"), 
       x = "x", y = "t")


### 基底関数と重みの関係 -----

# M個の基底関数を計算
phi_df <- Phi(x_vals, M) %>% # 基底関数
  dplyr::as_tibble(.name_repair = "unique") %>% # データフレームに変換
  dplyr::rename_with(.fn = ~paste0("m=", 0:(M-1)), .cols = 1:M) %>% # 列名を付与
  cbind(x = x_vals) %>% # x軸の値を結合
  tidyr::pivot_longer(
    cols = -x, names_to = "phi", names_transform = list(phi = as.factor), values_to = "phi_x"
  ) # long型に変換

# M個の基底関数を作図
ggplot() + 
  geom_line(data = phi_df, aes(x = x, y = phi_x, color = phi), 
            linetype = "dashed", size = 1) + # 基底関数
  labs(title = "Basis Function", 
       subtitle = paste0("mu=(", paste0(round(seq(0, 1, length.out = M-1), 2), collapse = ", "), ")"), 
       x = "x", y = "t", color = expression(phi[m](x)))


# M個の基底関数を重み付け
phi_w_df <- t(w_ml_m * t(Phi(x_vals, M))) %>% # 基底関数ごとに重み付け
  dplyr::as_tibble(.name_repair = "unique") %>% # データフレームに変換
  dplyr::rename_with(.fn = ~paste0("m=", 0:(M-1)), .cols = 1:M) %>% # 列名を付与
  cbind(x = x_vals) %>% # x軸の値を結合
  tidyr::pivot_longer(
    cols = -x, names_to = "phi", names_transform = list(phi = as.factor), values_to = "phi_x"
  ) # long型に変換

# 重み付けしたM個の基底関数を作図
ggplot() + 
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_line(data = ml_df, aes(x = x, y = t), color = "blue", size = 1) + # 推定したモデル
  geom_line(data = phi_w_df, aes(x = x, y = phi_x, color = phi), 
            linetype = "dashed", size = 1) + # 基底関数
  #ylim(c(-3, 3)) + # y軸の表示範囲
  labs(title = expression(t == sum(w[m] * phi[m](x), m == 0, M-1)), 
       subtitle = paste0("w=(", paste0(round(w_ml_m, 2), collapse = ", "), ")"), 
       x = "x", y = "t", color = expression(phi[m](x)))


