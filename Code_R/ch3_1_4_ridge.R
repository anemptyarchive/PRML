
# ch3.1.4：リッジ回帰 ----------------------------------------------------------

# 3.1.4項で利用するパッケージ
library(tidyverse)


# ・リッジ回帰 -------------------------------------------------------------------

### モデルの設定とデータの生成 -----

# ch3_1_1.Rを参照


### 最尤推定 -----

# 正則化係数を指定
lambda <- 0.5

# リッジ回帰の重みパラメータの最尤解を計算
w_ridge_m <- solve(lambda * diag(M) + t(phi_x_nm) %*% phi_x_nm) %*% t(phi_x_nm) %*% t_n %>% 
  as.vector()

# リッジ回帰の分散パラメータの最尤解を計算
sigma2_ridge <- sum((t_n - t(w_ridge_m) %*% t(phi_x_nm))^2) / N

# リッジ回帰の精度パラメータの最尤解を計算
beta_ridge <- 1 / sigma2_ridge


# 推定したパラメータによるモデルを計算
ridge_df <- tidyr::tibble(
  x = x_vec, # x軸の値
  t = t(w_ridge_m) %*% t(Phi(x, M)) %>% 
    as.vector(), # y軸の値
  minus_sigma = t - 2 * sqrt(sigma2_ridge), # μ - 2σ
  plus_sigma = t + 2 * sqrt(sigma2_ridge) # μ + 2σ
)

# 推定したパラメータによるモデルを作図
ggplot() + 
  geom_point(data = data_df, aes(x = x_n, y = t_n)) + # 観測データ
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_ribbon(data = model_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "cyan4", alpha = 0.1, color = "cyan4", linetype = "dotted") + # 真のノイズ範囲
  geom_line(data = ml_df, aes(x = x, y = t), color = "blue", linetype ="dashed") + # 推定したモデル:(正則化なし)
  geom_line(data = ridge_df, aes(x = x, y = t), color = "blue") + # 推定したモデル:(L2正則化)
  geom_ribbon(data = ridge_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "blue", alpha = 0.1, color = "blue", linetype = "dotted") + # 推定したノイズ範囲
  #ylim(c(-5, 5)) + 
  labs(title = "Ridge Regression", 
       subtitle = paste0("N=", N, ", M=", M, ", lambda=", lambda, ", beta=", round(beta_ridge, 2)), 
       x = "x", y = "t")



# ・重みパラメータと誤差関数の関係 -----------------------------------------------------------

### モデルの設定 -----

# 真の関数を指定
y_true <- function(x) {
  # 出力を計算
  y <- sin(2 * pi * x)
  return(y)
}

# 基底関数を指定
Phi <- function(x) {
  # 計算式を指定
  phi_x <- cbind(x^0, x^1)
  return(phi_x)
}


# パラメータの次元数を設定:(固定)
M <- 2

# 作図用のwの範囲を指定
w_i <- seq(-3, 3, by = 0.1)

# 作図用のwの点を作成
w_im <- cbind(
  rep(w_i, times = length(w_i)), # x軸の値
  rep(w_i, each = length(w_i)) # y軸の値
)


# データ数を指定
N <- 100

# (観測)データを生成
x_n <- runif(n = N, min = 0, max = 1) # 入力
t_n <- y_true(x_n) + rnorm(n = N, mean = 0, sd = 1) # 出力

# 基底関数により入力を変換
phi_x_nm <- Phi(x_n)


# 値を設定:(固定)
q <- 2

# 正則化項を計算
E_df <- tidyr::tibble(
  w_1 = w_im[, 1], # x軸の値
  w_2 = w_im[, 2], # y軸の値
  E_D = colSums((t_n - t(w_im %*% t(phi_x_nm)))^2) / N, # 二乗和誤差
  E_W = abs(w_1)^q + abs(w_2)^q # 正則化項
)

# 二乗和誤差関数の等高線図を作成
ggplot(E_df, aes(x = w_1, y = w_2)) + 
  geom_contour_filled(aes(x = w_1, y = w_2, z = E_D, fill = ..level..), alpha = 0.7) + # 二乗和誤差関数:(塗りつぶし)
  #geom_contour(aes(z = E_D), color = "blue") + # 二乗和誤差関数:(等高線)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E[D](w)), 
       subtitle = paste0("q=", q), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[D](w)))

# 正則化項の等高線図を作成
ggplot(E_df, aes(x = w_1, y = w_2)) + 
  geom_contour_filled(aes(x = w_1, y = w_2, z = E_W, fill = ..level..), alpha = 0.7) + # 正則化項:(塗りつぶし)
  #geom_contour(aes(z = E_W), color = "red") + # 正則化項:(等高線)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E[W](w)), 
       subtitle = paste0("q=", q), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[W](w)))


### 最尤推定 -----

# 正則化係数を指定
lambda <- 10

# リッジ回帰の重みパラメータの最尤解を計算
w_ridge_m <- solve(lambda * diag(M) + t(phi_x_nm) %*% phi_x_nm) %*% t(phi_x_nm) %*% t_n %>% 
  as.vector()

# 重みパラメータの最尤解を計算
w_ml_m <- solve(t(phi_x_nm) %*% phi_x_nm) %*% t(phi_x_nm) %*% t_n %>% 
  as.vector()

# 推定したパラメータによる誤差項を計算
E_D <- sum((t_n - w_ridge_m %*% t(Phi(x_n)))^2) / N
E_W <- abs(w_ridge_m[1])^q + abs(w_ridge_m[2])^q

# 重みパラメータをデータフレームに格納
w_df <- tidyr::tibble(
  w_1 = c(w_ridge_m[1], w_ml_m[1]), # x軸の値
  w_2 = c(w_ridge_m[2], w_ml_m[2]), # y軸の値
  method = factor(c("ridge", "ml"), levels = c(c("ridge", "ml"))) # ラベル
)

# 最尤解を作図
ggplot() + 
  geom_contour_filled(data = E_df, aes(x = w_1, y = w_2, z = E_D, fill = ..level..), alpha = 0.7) + # 二乗和誤差関数:(塗りつぶし)
  geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_D), color = "blue", breaks = E_D) + # 二乗和誤差関数:(等高線)
  geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_W), color = "red", breaks = E_W) + # 正則化項:(等高線)
  geom_point(data = w_df, aes(x = w_1, y = w_2, color = method), shape = 4, size = 5) + # パラメータの最尤解
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Ridge Regression", 
       subtitle = paste0("lambda=", lambda, 
                         ", w_ridge=(", paste0(round(w_ridge_m, 2), collapse = ", "), ")", 
                         ", w_ml=(", paste0(round(w_ml_m, 2), collapse = ", "), ")"), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[D](w)))


# 誤差関数を作図
ggplot(E_df, aes(x = w_1, y = w_2)) + 
  geom_contour_filled(aes(z = E_D + lambda * E_W, fill = ..level..)) + # 二乗和誤差関数:(塗りつぶし)
  geom_contour(aes(z = E_D), color = "blue", alpha = 0.5, linetype = "dashed") + # 二乗和誤差関数:(等高線)
  geom_contour(aes(z = E_W), color = "red", alpha = 0.5, linetype = "dashed") + # 正則化項
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E(w) == E[D](w) + lambda * E[w](w)), 
       subtitle = paste0("q=", q, ", lambda=", lambda), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E(w)))


### lambdaと最尤解の関係 ----

# 正則化係数の最大値を指定
lambda_max <- 100

# 使用するlambdaの値を作成
lambda_vec <- seq(0, lambda_max, by = 1)
length(lambda_vec)

# lambdaごとに最尤解を計算
anime_w_df <- tidyr::tibble()
anime_E_df <- tidyr::tibble()
for(lambda in lambda_vec) {
  # リッジ回帰の重みパラメータの最尤解を計算
  w_ridge_m <- solve(lambda * diag(M) + t(phi_x_nm) %*% phi_x_nm) %*% t(phi_x_nm) %*% t_n %>% 
    as.vector()
  
  # 推定したパラメータによる誤差項を計算
  tmp_E_D <- sum((t_n - w_ridge_m %*% t(Phi(x_n)))^2) / N # 二乗和誤差
  tmp_E_W <- abs(w_ridge_m[1])^q + abs(w_ridge_m[2])^q # 正則化項
  
  # アニメーション用のラベルを作成
  tmp_label <- paste0(
    "lambda=", lambda, ", E=", round(tmp_E_D + tmp_E_W, 2), ", E_D=", round(tmp_E_D, 2), ", E_W=", round(tmp_E_W, 2)
  )
  
  # 推定したパラメータをデータフレームに格納
  tmp_w_df <- tidyr::tibble(
    w_1 = c(w_ridge_m[1], w_ml_m[1]), # x軸の値
    w_2 = c(w_ridge_m[2], w_ml_m[2]), # y軸の値
    method = factor(c("ridge", "ml"), levels = c(c("ridge", "ml"))), # ラベル
    label = as.factor(tmp_label) # フレーム切替用のラベル
  )
  
  # 結果を結合
  anime_w_df <- rbind(anime_w_df, tmp_w_df)
  anime_E_df <- E_df %>% 
    dplyr::select(w_1, w_2, E_D) %>% # 使用する列を抽出
    cbind(label = as.factor(tmp_label)) %>% # フレーム切替用のラベル列を追加
    rbind(anime_E_df, .)
}


# 二乗和誤差と最尤解の関係を作図
anime_graph <- ggplot() + 
  geom_contour_filled(data = anime_E_df, aes(x = w_1, y = w_2, z = E_D, fill = ..level..), alpha = 0.7) + # 二乗和誤差関数:(塗りつぶし)
  geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_W), color = "red", breaks = 1) + # 正則化項
  geom_point(data = anime_w_df, aes(x = w_1, y = w_2, color = method), shape = 4, size = 5) + # パラメータの最尤解
  gganimate::transition_manual(label) + # フレーム
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Ridge Regression", 
       subtitle = paste0("{current_frame}"), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[D](w)))

# gif画像に変換
gganimate::animate(anime_graph, nframes = length(lambda_vec), fps = 10)


