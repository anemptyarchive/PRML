
# ch4.3.2-3 ロジスティック回帰 -----------------------------------------------------

# 4.3.2項で利用するパッケージ
library(tidyverse)


# 基底関数の準備 -------------------------------------------------------------------

# ロジスティックシグモイド関数を作成
sigmoid <- function(x) {
  # ロジスティックシグモイド関数の計算
  y <- 1 / (1 + exp(-x))
  return(y)
}

# シグモイド基底関数の計画行列を作成
phi <- function(x_n) {
  # データを標準化
  x_tilde_n <- (x_n - mean(x_n)) / sd(x_n)
  
  # マトリクスを初期化
  y_n <- matrix(1, nrow = length(x_n), ncol = 2)
  
  # ロジスティックシグモイド関数による変換
  y_n[, 2] <- sigmoid(x_tilde_n)
  return(y_n)
}

# 閾値の計算関数を作成:(シグモイド基底関数用)
threshold <- function(w_m, x_vals) {
  # 回帰式の逆関数
  x_tilde <- log(- w_m[1] / sum(w_m))
  
  # 標準化の逆関数
  x <- x_tilde * sd(x_vals) + mean(x_vals)
  return(x)
}


### モデルの設定 -----

# クラス割り当てのパラメータを指定:(クラス1となる確率)
pi <- 0.4

# データ生成のパラメータを指定:(クラス0, クラス1)
mu_k <- c(-1, 9)
sigma_k <- c(3, 3)

# データ数を指定
N <- 100

# 真のクラスを生成
t_n <- rbinom(n = N, size = 1, prob = pi)

# データを生成
x_n <- rnorm(n = N, mean = mu_k[t_n+1], sd = sigma_k[t_n+1])

# 観測データをデータフレームに格納
data_df <- tidyr::tibble(
  x = x_n, 
  t = t_n, 
  class = as.factor(t_n)
)

# 観測データのヒストグラムを作成
ggplot(data = data_df, aes(x = x, fill = class)) + 
  geom_histogram() + 
  labs(title = "Observation Data", 
       subtitle = paste0("pi=", pi, ", N0=", sum(t_n == 0), ", N1=", sum(t_n == 1)))


### ロジスティック回帰 -----

# 繰り返し回数を指定
max_iter <- 100

# 基底関数による変換
phi_x_nm <- phi(x_n)

# パラメータを初期化
w_logistic_m <- runif(n = 2, min = -10, max = 10)

# ニュートン-ラフソン法による推定
trace_w_mat <- t(w_logistic_m) # パラメータ
trace_E_vec <- NULL # 負の対数尤度
for(i in 1:max_iter) {
  # 重み付き和を計算
  a_n <- phi_x_nm %*% w_logistic_m %>% 
    as.vector()
  
  # ロジスティックシグモイド関数による変換
  y_n <- sigmoid(a_n)
  
  # 中間変数を計算
  r_nn <- diag(y_n)
  z_n <- phi_x_nm %*% w_logistic_m - solve(r_nn) %*% (y_n - t_n) %>% 
    as.vector()
  
  # パラメータを更新
  w_logistic_m <- solve(t(phi_x_nm) %*% r_nn %*% phi_x_nm) %*% t(phi_x_nm) %*% r_nn %*% z_n %>% 
    as.vector()
  
  # 推移を記録
  trace_w_mat <- rbind(trace_w_mat, w_logistic_m) # パラメータ
  trace_E_vec <- c(trace_E_vec, - sum(t_n * log(y_n) + (1 - t_n) * log(1 - y_n))) # 負の対数尤度
}


### 推定結果 -----

# x軸の範囲を指定
x_vals <- seq(min(x_n) - 10, max(x_n) + 10, by = 0.01)

# 推定したパラメータによるモデルを計算
logistic_df <- tidyr::tibble(
  x = x_vals, # 入力値
  a = phi(x_vals) %*% w_logistic_m %>% 
    as.vector(), # 変換した入力の重み付き和
  y = sigmoid(a) # 推定した確率値
)

# 推定したパラメータによるモデルを作図
ggplot() + 
  geom_point(data = data_df, aes(x = x, y = t, color = class)) + # 観測データ
  geom_line(data = logistic_df, aes(x = x, y = y)) + # 推定したモデル
  geom_vline(xintercept = threshold(w_logistic_m, x_vals), 
             color = "#00A968", linetype = "dashed") + # 分岐線
  labs(title = "Logistic Regression", 
       subtitle = paste0("iter:", max_iter, ", N=", N, 
                         ", threshold=", round(threshold(w_logistic_m, x_vals), 2), 
                         ", w=(", paste0(round(w_logistic_m, 2), collapse = ", "), ")"))


# 誤差の推移をデータフレームに格納
E_df <- tidyr::tibble(
  iteration = 0:(max_iter - 1), # 試行回数
  E = trace_E_vec # 負の対数尤度
)

# 誤差の推移を作図
ggplot(E_df, aes(x = iteration, y = E)) + 
  geom_line() + 
  labs(title = "Cross-Entropy Error", 
       y = expression(E(w)))


### 基底関数と重みの関係 -----

# M個の基底関数を重み付け
phi_w_df <- t(w_logistic_m * t(phi(x_vals))) %>% # 基底関数ごとに重み付け
  dplyr::as_tibble(.name_repair = "unique") %>% # データフレームに変換
  dplyr::rename_with(.fn = ~paste0("m=", 0:1), .cols = 1:2) %>% # 列名を付与
  cbind(x = x_vals) %>% # x軸の値を結合
  tidyr::pivot_longer(
    cols = -x, names_to = "phi", names_transform = list(phi = as.factor), values_to = "phi_x"
  ) # long型に変換

# 重み付けしたM個の基底関数を作図
ggplot() + 
  geom_line(data = logistic_df, aes(x = x, y = y), color = "red") + # 推定したモデル
  geom_line(data = logistic_df, aes(x = x, y = a), color = "orange", size = 1) + # 重み付き和
  geom_line(data = phi_w_df, aes(x = x, y = phi_x, color = phi), linetype = "dashed", size = 1) + # 基底関数
  geom_point(data = data_df, aes(x = x, y = t)) + # 観測データ
  #ylim(c(-10, 10)) + # y軸の表示範囲
  labs(title = expression(a == w[0] + w[1] * phi[1](x)), 
       subtitle = paste0("w=(", paste0(round(w_logistic_m, 2), collapse = ", "), ")"), 
       x = "x", y = "a", color = expression(phi[m](x)))


### 誤差関数のグラフとパラメータの推移 -----

# 作図用のwの範囲を指定
w_vals <- seq(-100, 100, by = 1)

# 作図用のwの点を作成
w_im <- expand.grid(w_vals, w_vals) %>% 
  as.matrix()

# 出力を計算
y_ni <- sigmoid(phi_x_nm %*% t(w_im))

# 交差エントロピー誤差を計算
E_i <- - colSums(log(y_ni^t_n) + log((1 - y_ni)^(1 - t_n)))

# 誤差関数をデータフレームに格納
w_df <- tidyr::tibble(
  w_0 = w_im[, 1], # x軸の値
  w_1 = w_im[, 2], # y軸の値
  E = E_i
)

# パラメータの推移をデータフレームに格納
trace_w_df <- trace_w_mat %>% 
  dplyr::as_tibble() %>% # データフレームに変換
  dplyr::rename_with(.fn = ~paste0("w_", 0:1), .cols = 1:2) %>% # 列名を付与
  dplyr::mutate(iteration = 0:max_iter) # 試行回数を結合

# パラメータの推移を作図
ggplot() + 
  #geom_contour(data = w_df, aes(x = w_0, y = w_1, z = E, color = ..level..)) + # 誤差関数
  geom_contour(data = w_df, aes(x = w_0, y = w_1, z = E, color = ..level..), 
               breaks = seq(0, 100, length.out = 10)) + # 誤差関数:(等高線を引く値を指定)
  geom_point(data = trace_w_df, aes(x = w_0, y = w_1), color = "#00A968") + # パラメータの推移:(点)
  geom_line(data = trace_w_df, aes(x = w_0, y = w_1), color = "#00A968") + # パラメータの推移:(線)
  labs(title = "Cross-Entropy Error", 
       subtitle = paste0("iter:", max_iter, ", N=", N), 
       x = expression(w[0]), y = expression(w[1]), color = expression(E(w)))


### モデルの推移をアニメーションで確認 -----

# 最終的な誤差を計算
y_n <- sigmoid(as.vector(phi_x_nm %*% w_logistic_m))
trace_E_vec <- c(trace_E_vec, - sum(t_n * log(y_n) + (1 - t_n) * log(1 - y_n)))

# 試行ごとのモデルを計算
trace_logistic_df <- tidyr::tibble()
for(i in 0:max_iter) {
  # i回目の値を取得
  tmp_w_m <- trace_w_mat[i+1, ]
  tmp_E <- trace_E_vec[i+1]
  
  # i回目のパラメータによるモデルを計算
  tmp_logistic_df <- tidyr::tibble(
    x = x_vals, 
    y = sigmoid(as.vector(phi(x_vals) %*% tmp_w_m)), 
    threshold_x = threshold(tmp_w_m, x_vals), # y=0.5となるxの値
    label = paste0(
      "iter:", i, ", E=", round(tmp_E, 2), 
      ", threshold=", round(threshold(tmp_w_m, x_vals), 2), 
      ", w=(", paste0(round(tmp_w_m, 2), collapse = ", "), ")"
    ) %>% 
      as.factor() # フレーム切替用のラベル
  )
  
  # 結果を結合
  trace_logistic_df <- rbind(trace_logistic_df, tmp_logistic_df)
}

# 推定したパラメータによるモデルを作図
anime_graph <- ggplot() + 
  geom_point(data = data_df, aes(x = x, y = t, color = class)) + # 観測データ
  geom_line(data = trace_logistic_df, aes(x = x, y = y)) + # 推定したモデル
  geom_vline(data = trace_logistic_df, aes(xintercept = threshold_x), 
             color = "#00A968", linetype = "dashed") + # 閾値
  gganimate::transition_manual(label) + # フレーム
  xlim(c(min(x_vals), max(x_vals))) + # x軸の表示範囲
  labs(title = "Logistic Regression", 
       subtitle = "{current_frame}")

# gif画像に変換
gganimate::animate(anime_graph, nframe = max_iter + 1, fps = 10)


