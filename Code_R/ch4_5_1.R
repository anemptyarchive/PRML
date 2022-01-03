
# ch4.5.1 ベイズロジスティック回帰 ----------------------------------------------------

# 4.5.1項で利用するパッケージ
library(tidyverse)
library(mvnfast)
library(gganimate)


### モデルの設定とデータの生成 -----

# ch4_3_2を参照


### ベイズロジスティック回帰 -----

# パラメータの次元数(基底関数の数)を設定:(固定)
M <- 2

# 事前分布のパラメータの初期値を指定
m_m <- c(0, 0) # 平均
s_mm <- matrix(c(10, 0, 0, 10), nrow = M, ncol = M) # 分散共分散行列
s_inv_mm <- solve(s_mm) # 精度行列

# モデルのパラメータを初期化
w_logistic_m <- runif(n = M, min = -1, max = 1)

# 試行回数を指定
max_iter_laplace <- 500
max_iter_newton <- 100

# 推移の確認用の受け皿を作成
trace_w_mat <- matrix(NA, nrow = max_iter_laplace, ncol = M) # モデルの重みパラメータ
trace_s_arr <- array(NA, dim = c(M, M, max_iter_laplace)) # 事前分布の分散共分散パラメータ
trace_E_vec <- rep(NA, times = max_iter_laplace) # 負の対数事後確率

# ラプラス近似
for(i in 1:max_iter_laplace) {
  
  # ニュートン-ラフソン法による推定
  for(j in 1:max_iter_newton) {
    # 重み付き和を計算
    a_n <- (phi_x_nm %*% w_logistic_m) %>% 
      as.vector()
    
    # ロジスティックシグモイド関数による変換
    y_n <- sigmoid(a_n)
    
    # 中間変数を作成
    r_nn <- diag(y_n)
    
    # 負の対数事後分布の勾配を計算
    nabla_E_m <- (t(phi_x_nm) %*% (y_n - t_n) + s_inv_mm %*% (w_logistic_m - m_m)) %>% 
      as.vector()
    
    # 負の対数事後分布のヘッセ行列を計算
    h_mm <- s_inv_mm + t(phi_x_nm) %*% r_nn %*% phi_x_nm
    
    # 重みパラメータを更新
    w_logistic_m <- (w_logistic_m - solve(h_mm) %*% nabla_E_m) %>% 
      as.vector()
  }
  
  # 事前分布のパラメータを更新
  m_m <- w_logistic_m
  s_inv_mm <- h_mm
  s_mm <- solve(h_mm)
  
  # 負の対数事後確率を計算
  y_n <- sigmoid(as.vector(phi_x_nm %*% w_logistic_m))
  negative_log_likelihood <- -sum(dbinom(x = t_n, size = 1, prob = y_n, log = TRUE)) # 負の対数尤度関数
  negative_log_prior <- -mvnfast::dmvn(X = w_logistic_m, mu = m_m, sigma = s_mm, log = TRUE) # 負の対数事前確率
  
  # 値を記録
  trace_w_mat[i, ] <- w_logistic_m # モデルの重みパラメータ
  trace_s_arr[, , i] <- s_mm # 事前分布の分散共分散パラメータ
  trace_E_vec[i] <- negative_log_likelihood + negative_log_prior # 負の対数事後確率
  
  # 途中経過を表示
  message("\r", "iter:", i, " (", round(i / max_iter_laplace * 100, 2), "%)", ", E=", trace_E_vec[i], appendLF = FALSE)
}


### 推定したパラメータによるモデル -----

# 作図用のxの点を作成
x_vals <- seq(min(x_n) - 10, max(x_n) + 10, length.out = 1000)

# MAP解によるモデルを計算
logistic_df <- tidyr::tibble(
  x = x_vals, # 入力値
  a = phi(x_vals) %*% w_logistic_m %>% 
    as.vector(), # 変換した入力の重み付き和
  y = sigmoid(a) # 推定した確率値
)

# MAP解によるモデルを作図
ggplot() + 
  geom_point(data = data_df, aes(x = x, y = t, color = class)) + # 観測データ
  geom_line(data = logistic_df, aes(x = x, y = y)) + # 推定したモデル
  geom_vline(xintercept = threshold(w_logistic_m, x_vals), 
             color = "#00A968", linetype = "dashed") + # クラスの閾値
  labs(title = "Bayesian Logistic Regression", 
       subtitle = paste0("iter:", max_iter_laplace, ", N=", N, 
                         ", threshold=", round(threshold(w_logistic_m, x_vals), 2), 
                         ", w=(", paste0(round(w_logistic_m, 2), collapse = ", "), ")"))


# フレームに利用する試行番号を指定
i_vec <- seq(1, max_iter_laplace, by = 10)
length(i_vec) # フレーム数

# 試行ごとのモデルを計算
trace_logistic_df <- tidyr::tibble()
for(i in i_vec) {
  # i回目の値を取得
  tmp_w_m <- trace_w_mat[i, ]
  
  # i回目のパラメータによるモデルを計算
  tmp_logistic_df <- tidyr::tibble(
    x = x_vals, 
    y = sigmoid(as.vector(phi(x_vals) %*% tmp_w_m)), 
    threshold_x = threshold(tmp_w_m, x_vals), # y=0.5となるxの値
    label = paste0(
      "iter:", i, ", E=", round(trace_E_vec[i], 2), 
      ", threshold=", round(threshold(tmp_w_m, x_vals), 2), 
      ", w=(", paste0(round(tmp_w_m, 2), collapse = ", "), ")"
    ) %>% 
      as.factor() # フレーム切替用のラベル
  )
  
  # 結果を結合
  trace_logistic_df <- rbind(trace_logistic_df, tmp_logistic_df)
  
  # 途中経過を表示
  message("\r", "iter:", i, " (", round(i / max_iter_laplace * 100, 2), "%)", appendLF = "FALSE")
}

# MAP解によるモデルを作図
anime_model_graph <- ggplot() + 
  geom_point(data = data_df, aes(x = x, y = t, color = class)) + # 観測データ
  geom_line(data = trace_logistic_df, aes(x = x, y = y)) + # 推定したモデル
  geom_vline(data = trace_logistic_df, aes(xintercept = threshold_x), 
             color = "#00A968", linetype = "dashed") + # クラスの閾値
  gganimate::transition_manual(label) + # フレーム
  xlim(c(min(x_vals), max(x_vals))) + # x軸の表示範囲
  labs(title = "Bayesian Logistic Regression", 
       subtitle = "{current_frame}")

# gif画像に変換
gganimate::animate(anime_model_graph, nframe = length(i_vec), fps = 10)


### 重みパラメータの近似事後分布 -----

# 作図用のwの点を作成
w_im <- expand.grid(
  seq(m_m[1] - 3 * sqrt(s_mm[1, 1]), m_m[1] + 3 * sqrt(s_mm[1, 1]), length.out = 500), # x軸の値
  seq(m_m[2] - 3 * sqrt(s_mm[2, 2]), m_m[2] + 3 * sqrt(s_mm[2, 2]), length.out = 500)  # y軸の値
) %>% 
  as.matrix()

# 重みパラメータの近似事後分布を計算
posterior_df <- tidyr::tibble(
  w_0 = w_im[, 1], # x軸の値
  w_1 = w_im[, 2], # y軸の値
  density = mvnfast::dmvn(X = w_im, mu = m_m, sigma = s_mm) # 確率密度
)

# 重みパラメータのMAP解を格納
w_df <- tidyr::tibble(
  w_0 = w_logistic_m[1], # x軸の値
  w_1 = w_logistic_m[2]  # y軸の値
)

# 重みパラメータの近似事後分布を作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x = w_0, y = w_1, z = density, color = ..level..)) + # 近似事後分布
  geom_point(data = w_df, aes(x = w_0, y = w_1), color = "red", shape = 4, size = 5) + # MAP解
  labs(title = "Approximate Posterior Distribution", 
       subtitle = paste0("iter:", max_iter_laplace, 
                         ", w=(", paste0(round(w_logistic_m, 2), collapse = ", "), ")", 
                         ", s=(", paste0(round(s_mm, 5), collapse = ", "), ")"), 
       x = expression(w[0]), y = expression(w[1]), color = "density")


# フレームに利用する試行回数を指定
i_vec <- seq(250, max_iter_laplace, by = 10)
length(i_vec) # フレーム数

# 試行ごとに近似事後分布を計算
trace_w_df <- tidyr::tibble()
trace_posterior_df <- tidyr::tibble()
for(i in i_vec) {
  # i回目のパラメータを取得
  tmp_w_m <- trace_w_mat[i, ]
  tmp_m_m <- trace_w_mat[i, ]
  tmp_s_mm <- trace_s_arr[, , i]
  
  # フレーム切替用のラベルを作成
  label_text <- paste0(
    "iter:", i, ", E=", round(trace_E_vec[i], 2), 
    ", w=(", paste0(round(tmp_w_m, 2), collapse = ", "), ")", 
    ", s=(", paste0(round(tmp_s_mm, 5), collapse = ", "), ")"
  )
  
  # i回目の重みパラメータを格納
  tmp_w_df <- tidyr::tibble(
    w_0 = tmp_w_m[1], # x軸の値
    w_1 = tmp_w_m[2], # y軸の値
    label = as.factor(label_text) # フレーム切替用のラベル
  )
  
  # 作図用のwの点を作成
  w_im <- expand.grid(
    seq(tmp_m_m[1] - 3 * sqrt(tmp_s_mm[1, 1]), tmp_m_m[1] + 3 * sqrt(tmp_s_mm[1, 1]), length.out = 500), # x軸の値
    seq(tmp_m_m[2] - 3 * sqrt(tmp_s_mm[2, 2]), tmp_m_m[2] + 3 * sqrt(tmp_s_mm[2, 2]), length.out = 500)  # y軸の値
  ) %>% 
    as.matrix()
  
  # i回目の近似事後分布を計算
  tmp_posterior_df <- tidyr::tibble(
    w_0 = w_im[, 1], # x軸の値
    w_1 = w_im[, 2], # y軸の値
    density = mvnfast::dmvn(X = w_im, mu = tmp_m_m, sigma = tmp_s_mm), # 確率密度
    label = as.factor(label_text) # フレーム切替用のラベル
  )
  
  # 結果を結合
  trace_w_df <- rbind(trace_w_df, tmp_w_df)
  trace_posterior_df <- rbind(trace_posterior_df, tmp_posterior_df)
  
  # 途中経過を表示
  message("\r", "iter:", i, " (", round(i / max_iter_laplace * 100, 2), "%)", appendLF = "FALSE")
}

# 重みパラメータの近似事後分布を作図
anime_posterior_graph <- ggplot() + 
  geom_contour(data = trace_posterior_df, aes(x = w_0, y = w_1, z = density, color = ..level..)) + # 近似事後分布
  geom_point(data = trace_w_df, aes(x = w_0, y = w_1), color = "red", shape = 4, size = 5) + # MAP解
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Approximate Posterior Distribution", 
       subtitle = "{current_frame}", 
       x = expression(w[0]), y = expression(w[1]))

# gif画像に変換
gganimate::animate(anime_posterior_graph, nframe = length(i_vec), fps = 10)


### 負の対数事後確率の推移 -----

# 負の対数事後確率の推移を格納
E_df <- tidyr::tibble(
  iteration = 1:max_iter_laplace, 
  E = trace_E_vec
)

# 負の対数事後確率の推移を作図
ggplot(E_df, aes(x = iteration, y = E)) + 
  geom_line() + 
  labs(title = "Negative log Posterior", 
       y = expression(E(w)))


### パラメータの推移 -----

# モデルの重みパラメータの推移を格納
trace_w_df <- trace_w_mat %>% 
  tidyr::as_tibble() %>% # データフレームに変換
  dplyr::rename_with(.fn = ~0:(M-1), .cols = 1:M) %>% # 列名を付与
  dplyr::mutate(iteration = 1:max_iter_laplace) %>%  # 試行番号列を結合
  tidyr::pivot_longer(cols = !iteration, names_to = "m", values_to = "value") # 縦型に変換

# モデルの重みパラメータの推移を作図
ggplot(trace_w_df, aes(x = iteration, y = value, color = m)) + 
  geom_line() + 
  labs(title = "w = m", 
       y = expression(w[m]))


# 事前分布の分散共分散パラメータの推移を格納
trace_s_df <- trace_s_arr %>% 
  as.vector() %>% # ベクトルに変換
  matrix(nrow = max_iter_laplace, ncol = M^2, byrow = TRUE) %>% # マトリクスに変換
  dplyr::as_tibble() %>% # データフレームに変換
  dplyr::rename_with(.fn = ~paste0("s_", rep(1:M, times = M), rep(1:M, each = M))) %>% # 列名を付与
  dplyr::mutate(iteration = 1:max_iter_laplace) %>% # 試行番号列を結合
  tidyr::pivot_longer(cols = !iteration, names_to = "idx", values_to = "value") # 縦型に変換

# 事前分布の分散共分散パラメータの推移を作図
ggplot(trace_s_df, aes(x = iteration, y = value, color = idx)) + 
  geom_line() + 
  #ylim(c(-0.5, 0.5)) + # y軸の表示範囲
  labs(title = "S", 
       y = expression(s[mm]))


### 負の対数事後確率とパラメータの推移 -----

# 作図用のwの点を作成
w_im <- expand.grid(
  seq(min(trace_w_mat[, 1]) - 5, max(trace_w_mat[, 1]) + 5, length.out = 500), 
  seq(min(trace_w_mat[, 2]) - 5, max(trace_w_mat[, 2]) + 5, length.out = 500)
) %>% 
  as.matrix()

# 出力を計算
y_ni <- sigmoid(phi_x_nm %*% t(w_im))

# 負の対数事後確率を計算
negative_log_likelihood_i <- -colSums(log(y_ni^t_n) + log((1 - y_ni)^(1 - t_n))) # 負の対数尤度関数
negative_log_prior_i <- -mvnfast::dmvn(X = w_im, mu = m_m, sigma = s_mm, log = TRUE) # 負の対数事前確率

# 負の対数事後確率を格納
error_df <- tidyr::tibble(
  w_0 = w_im[, 1], # x軸の値
  w_1 = w_im[, 2], # y軸の値
  E = negative_log_likelihood_i + negative_log_prior_i
)

# パラメータの推移をデータフレームに格納
trace_w_df <- trace_w_mat %>% 
  dplyr::as_tibble() %>% # データフレームに変換
  dplyr::rename_with(.fn = ~paste0("w_", 0:1), .cols = 1:M) %>% # 列名を付与
  dplyr::mutate(iteration = 1:max_iter_laplace) # 試行回数を結合

# パラメータの推移を作図
ggplot() + 
  geom_contour(data = error_df, aes(x = w_0, y = w_1, z = E, color = ..level..)) + # 誤差関数
  #geom_contour(data = error_df, aes(x = w_0, y = w_1, z = E, color = ..level..), 
  #             breaks = seq(0, max(error_df[["E"]]) * 0.1, length.out = 10)) + # 誤差関数:(等高線を引く値を指定)
  geom_point(data = trace_w_df, aes(x = w_0, y = w_1), color = "#00A968") + # パラメータの推移
  labs(title = "Negative Log Posterior", 
       subtitle = paste0("iter:", max_iter_laplace), 
       x = expression(w[0]), y = expression(w[1]), color = expression(E(w)))


# フレームに利用する試行回数を間引く
i_vec <- seq(1, max_iter_laplace, by = 50)
length(i_vec) # フレーム数

# 試行ごとに近似事後分布を計算
trace_w_df <- tidyr::tibble()
trace_error_df <- tidyr::tibble()
for(i in i_vec) {
  # i回目のパラメータを取得
  tmp_w_m <- trace_w_mat[i, ]
  tmp_m_m <- trace_w_mat[i, ]
  tmp_s_mm <- trace_s_arr[, , i]
  
  # フレーム切替用のラベルを作成
  label_text <- paste0(
    "iter:", i, ", E=", round(trace_E_vec[i], 2), 
    ", w=(", paste0(round(tmp_w_m, 2), collapse = ", "), ")"
  )
  
  # i回目の重みパラメータを格納
  tmp_w_df <- tidyr::tibble(
    w_0 = trace_w_mat[1:i, 1], # x軸の値
    w_1 = trace_w_mat[1:i, 2], # y軸の値
    label = as.factor(label_text) # フレーム切替用のラベル
  )

  # 負の対数事後確率を計算
  y_ni <- sigmoid(phi_x_nm %*% t(w_im))
  negative_log_likelihood_i <- -colSums(log(y_ni^t_n) + log((1 - y_ni)^(1 - t_n))) # 負の対数尤度関数
  negative_log_prior_i <- -mvnfast::dmvn(X = w_im, mu = tmp_m_m, sigma = tmp_s_mm, log = TRUE) # 負の対数事前確率
  
  # i回目の負の対数事後確率を計算
  tmp_error_df <- tidyr::tibble(
    w_0 = w_im[, 1], # x軸の値
    w_1 = w_im[, 2], # y軸の値
    E = negative_log_likelihood_i + negative_log_prior_i, # 負の対数事後確率
    label = as.factor(label_text) # フレーム切替用のラベル
  )
  
  # 結果を結合
  trace_w_df <- rbind(trace_w_df, tmp_w_df)
  trace_error_df <- rbind(trace_error_df, tmp_error_df)
  
  # 途中経過を表示
  message("\r", "iter:", i, " (", round(i / max_iter_laplace * 100, 2), "%)", appendLF = "FALSE")
}

# パラメータの推移を作図
anime_error_graph <- ggplot() + 
  geom_contour(data = trace_error_df, aes(x = w_0, y = w_1, z = E, color = ..level..)) + # 対数事後確率
  #geom_contour(data = trace_error_df, aes(x = w_0, y = w_1, z = E, color = ..level..), 
  #             breaks = seq(0, max(trace_error_df[["E"]]) * 0.1, length.out = 10)) + # 誤差関数:(等高線を引く値を指定)
  geom_point(data = trace_w_df, aes(x = w_0, y = w_1), color = "#00A968") + # パラメータの推移
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Negative Log Posterior", 
       subtitle = "{current_frame}", 
       x = expression(w[0]), y = expression(w[1]), color = expression(E(w)))

# gif画像に変換
gganimate::animate(anime_error_graph, nframe = length(i_vec), fps = 10)


