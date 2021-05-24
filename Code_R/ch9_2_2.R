
# 9.2.2 混合ガウス分布のEMアルゴリズム --------------------------------------------------

# 9.2.2項で利用するパッケージ
library(tidyverse)
library(mvnfast)


### 真の分布の設定 -----

# 次元数を設定:(固定)
D <- 2

# クラスタ数を指定
K <- 3

# K個の真の平均を指定
mu_truth_kd <- matrix(
  c(5, 35, 
    -20, -10, 
    30, -20), nrow = K, ncol = D, byrow = TRUE
)

# K個の真の共分散行列を指定
sigma2_truth_ddk <- array(
  c(250, 65, 65, 270, 
    125, -45, -45, 175, 
    210, -15, -15, 250), dim = c(D, D, K)
)

# 真の混合係数を指定
pi_truth_k <- c(0.45, 0.25, 0.3)


# 作図用のx軸のxの値を作成
x_1_vec <- seq(
  min(mu_truth_kd[, 1] - 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  max(mu_truth_kd[, 1] + 3 * sqrt(sigma2_truth_ddk[1, 1, ])), 
  length.out = 300)

# 作図用のy軸のxの値を作成
x_2_vec <- seq(
  min(mu_truth_kd[, 2] - 3 * sqrt(sigma2_truth_ddk[2, 2, ])), 
  max(mu_truth_kd[, 2] + 3 * sqrt(sigma2_truth_ddk[2, 2, ])), 
  length.out = 300
)

# 作図用のxの点を作成
x_point_mat <- cbind(
  rep(x_1_vec, times = length(x_2_vec)), 
  rep(x_2_vec, each = length(x_1_vec))
)


# 真の分布を計算
model_dens <- 0
for(k in 1:K) {
  # クラスタkの分布の確率密度を計算
  tmp_dens <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_truth_kd[k, ], sigma = sigma2_truth_ddk[, , k]
  )
  
  # K個の分布を線形結合
  model_dens <- model_dens + pi_truth_k[k] * tmp_dens
}

# 真の分布をデータフレームに格納
model_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = model_dens
)

# 真の分布を作図
ggplot(model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + 
  geom_contour() + # 真の分布
  labs(title = "Mixture of Gaussians", 
       subtitle = paste0("K=", K), 
       x = expression(x[1]), y = expression(x[2]))


### 観測データの生成 -----

# (観測)データ数を指定
N <- 250

# 潜在変数を生成
z_truth_nk <- rmultinom(n = N, size = 1, prob = pi_truth_k) %>% 
  t()

# クラスタ番号を取得
z_truth_n <- which(t(z_truth_nk) == 1, arr.ind = TRUE) %>% 
  .[, "row"]

# (観測)データを生成
x_nd <- matrix(0, nrow = N, ncol = D)
for(n in 1:N) {
  # n番目のデータのクラスタ番号を取得
  k <- z_truth_n[n]
  
  # n番目のデータを生成
  x_nd[n, ] = mvnfast::rmvn(n = 1, mu = mu_truth_kd[k, ], sigma = sigma2_truth_ddk[, , k])
}

# 観測データと真のクラスタ番号をデータフレームに格納
data_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2], 
  cluster = as.factor(z_truth_n)
)

# 観測データの散布図を作成
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               linetype = "dashed") + # 真の分布
  #geom_contour_filled(data = model_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), 
  #                    alpha = 0.6, linetype = "dashed") + # 真の分布:(塗りつぶし)
  geom_point(data = data_df, aes(x = x_n1, y = x_n2, color = cluster)) + # 真のクラスタ
  labs(title = "Mixture of Gaussians", 
       subtitle = paste0('N=', N, ", pi=(", paste0(pi_truth_k, collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


### 初期値の設定 -----

# 平均パラメータの初期値を生成
mu_kd <- matrix(0, nrow = K, ncol = D)
for(d in 1:D) {
  mu_kd[, d] <- runif(n = K, min = min(x_nd[, d]), max = max(x_nd[, d]))
}

# 共分散行列の初期値を指定
sigma2_ddk <- (diag(D) * 1000) %>% # 値を指定
  rep(times = K) %>% # 値を複製
  array(dim = c(D, D, K)) # 配列を整形

# 混合係数の初期値を生成
pi_k <- runif(n = K, min = 0, max = 1)
pi_k <- pi_k / sum(pi_k) # 正規化


# 初期値による混合分布を計算
init_dens <- 0
for(k in 1:K) {
  # クラスタkの分布の確率密度を計算
  tmp_dens <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_kd[k, ], sigma = sigma2_ddk[, , k]
  )
  
  # K個の分布を線形結合
  init_dens <- init_dens + pi_k[k] * tmp_dens
}

# 初期値による分布をデータフレームに格納
init_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = init_dens
)

# 初期値による分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_contour(data = init_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 推定値による分布
  labs(title = "Mixture of Gaussians", 
       subtitle = paste0("iter:", 0, ", K=", K), 
       x = expression(x[1]), y = expression(x[2]))


# 初期値による対数尤度を計算:式(9.14)
term_dens_nk <- matrix(0, nrow = N, ncol = K)
for(k in 1:K) {
  # クラスタkの混合分布の確率密度を計算:式(9.14)の波括弧
  tmp_N_dens_n <- mvnfast::dmvn(
    X = x_nd, mu = mu_kd[k, ], sigma = sigma2_ddk[, , k]
  )
  term_dens_nk[, k] <- pi_k[k] * tmp_N_dens_n
}
L <- sum(log(rowSums(term_dens_nk)))


### 推論処理 -----

# 試行回数を指定
MaxIter <- 100


# 計算に用いる中間変数を作成
tmp_gamma_nk <- matrix(0, nrow = N, ncol = K)

# 推移の確認用の受け皿を作成
trace_L_i         <- rep(0, MaxIter + 1)
trace_gamma_ink   <- array(0, dim = c(MaxIter + 1, N, K))
trace_mu_ikd      <- array(0, dim = c(MaxIter + 1, K, D))
trace_sigma2_iddk <- array(0, dim = c(MaxIter + 1, D, D, K))
trace_pi_ik       <- matrix(0, nrow = MaxIter + 1, ncol = K)

# 初期値を記録
trace_L_i[1] <- L
trace_gamma_ink[1, , ] <- NA
trace_mu_ikd[1, , ]        <- mu_kd
trace_sigma2_iddk[1, , , ] <- sigma2_ddk
trace_pi_ik[1, ]           <- pi_k

# 最尤推定
for(i in 1:MaxIter) {
  
  # 負担率を計算:式(9.13)
  for(k in 1:K) {
    # 正規化前の負担率を計算:式(9.13)の分子
    tmp_N_dens <- mvnfast::dmvn(
      X = x_nd, mu = mu_kd[k, ], sigma = sigma2_ddk[, , k]
    )
    tmp_gamma_nk[, k] <- pi_k[k] * tmp_N_dens
  }
  gamma_nk <- tmp_gamma_nk / rowSums(tmp_gamma_nk) # 正規化
  
  # 各クラスタとなるデータ数の期待値を計算:(9.18)
  N_k <- colSums(gamma_nk)
  
  for(k in 1:K) {
    # 平均パラメータの最尤解を計算:式(9.17)
    mu_kd[k, ] <- (t(gamma_nk[, k]) %*% x_nd / N_k[k]) %>% 
      as.vector()
    
    # 共分散行列の最尤解を計算:(9.19)
    term_x_dn <- t(x_nd) - mu_kd[k, ]
    sigma2_ddk[, , k] <- term_x_dn %*% (gamma_nk[, k] * t(term_x_dn)) / N_k[k]
  }
  
  # 混合係数の最尤解を計算:式(9.22)
  pi_k <- N_k / N
  
  # 対数尤度を計算:式(9.14)
  for(k in 1:K) {
    # クラスタkの混合分布の確率密度を計算:式(9.14)の波括弧
    tmp_N_dens_n <- mvnfast::dmvn(
      X = x_nd, mu = mu_kd[k, ], sigma = sigma2_ddk[, , k]
    )
    term_dens_nk[, k] <- pi_k[k] * tmp_N_dens_n
  }
  L <- sum(log(rowSums(term_dens_nk)))
  
  # i回目の結果を記録
  trace_L_i[i + 1] <- L
  trace_gamma_ink[i + 1, , ] <- gamma_nk
  trace_mu_ikd[i + 1, , ]        <- mu_kd
  trace_sigma2_iddk[i + 1, , , ] <- sigma2_ddk
  trace_pi_ik[i + 1, ]           <- pi_k
  
  # 動作確認
  print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### 推論結果の確認 -----

# 真の平均をデータフレームに格納
mu_df <- tibble(
  x_1 = mu_truth_kd[, 1], 
  x_2 = mu_truth_kd[, 2]
)


# 最後の更新値による混合分布を計算
res_dens <- 0
for(k in 1:K) {
  # クラスタkの分布の確率密度を計算
  tmp_dens <- mvnfast::dmvn(
    X = x_point_mat, mu = mu_kd[k, ], sigma = sigma2_ddk[, , k]
  )
  
  # K個の分布を線形結合
  res_dens <- res_dens + pi_k[k] * tmp_dens
}

# 最終的な分布をデータフレームに格納
res_df <- tibble(
  x_1 = x_point_mat[, 1], 
  x_2 = x_point_mat[, 2], 
  density = res_dens
)

# 最終的な分布を作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = res_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 推定値による分布
  labs(title = "Mixture of Gaussians:Maximum Likelihood", 
       subtitle = paste0("iter:", MaxIter, 
                         ", L=", round(L, 1), 
                         ", N=", N, 
                         ", pi=(", paste0(round(pi_k, 3), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


# 観測データと負担率が最大のクラスタ番号をデータフレームに格納
gamma_df <- tibble(
  x_n1 = x_nd[, 1], 
  x_n2 = x_nd[, 2], 
  cluster = as.factor(max.col(gamma_nk)), # 負担率が最大のクラスタ番号
  prob = gamma_nk[cbind(1:N, max.col(gamma_nk))] # 負担率の最大値
)

# 最終的なクラスタを作図
ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = res_df, aes(x = x_1, y = x_2, z = density, fill = ..level..), 
                      alpha = 0.6) + # 推定値による分布
  geom_point(data = gamma_df, aes(x = x_n1, y = x_n2, color = cluster), 
             alpha = gamma_df[["prob"]]) + # 負担率によるクラスタ
  labs(title = "Mixture of Gaussians:Maximum Likelihood", 
       subtitle = paste0("iter:", MaxIter, 
                         ", L=", round(L, 1), 
                         ", N=", N, 
                         ", pi=(", paste0(round(pi_k, 3), collapse = ", "), ")"), 
       x = expression(x[1]), y = expression(x[2]))


# 対数尤度の推移をデータフレームに格納
L_df <- tibble(
  iteration = 0:MaxIter, 
  L = trace_L_i
)

# 対数尤度の推移を作図
ggplot(L_df, aes(x = iteration, y = L)) + 
  geom_line() + 
  labs(title = "Maximum Likelihood", 
       subtitle = "Log Likelihood")


### パラメータの更新値の推移を確認 -----

# 平均パラメータの推移を作図
dplyr::as_tibble(trace_mu_ikd) %>% # データフレームに変換
  magrittr::set_names(
    paste0(rep(paste0("k=", 1:K), D), rep(paste0(", d=", 1:D), each = K))
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "dim", # 現列名を格納する列名
    values_to = "value" # 現要素を格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
  geom_line() + 
  geom_hline(yintercept = as.vector(mu_truth_kd), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Maximum Likelihood", 
       subtitle = expression(mu))

# 共分散行列の推移を作図
dplyr::as_tibble(trace_sigma2_iddk) %>% # データフレームに変換
  magrittr::set_names(
    paste0(
      rep(paste0("d=", 1:D), times = D * K), 
      rep(rep(paste0(", d=", 1:D), each = D), times = K), 
      rep(paste0(", k=", 1:K), each = D * D)
    )
  ) %>% # 列名として次元情報を付与
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "dim", # 現列名を格納する列名
    values_to = "value" # 現要素を格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = dim)) + 
  geom_line(alpha = 0.5) + 
  geom_hline(yintercept = as.vector(sigma2_truth_ddk), 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Maximum Likelihood", 
       subtitle = expression(Sigma))

# 混合係数の推移を作図
dplyr::as_tibble(trace_pi_ik) %>% # データフレームに変換
  cbind(iteration = 1:(MaxIter + 1)) %>% # 試行回数の列を追加
  tidyr::pivot_longer(
    cols = -iteration, # 変換しない列
    names_to = "cluster", # 現列名を格納する列名
    names_prefix = "V", # 現列名の頭から取り除く文字列
    names_ptypes = list(cluster = factor()), # 現列名を値とする際の型
    values_to = "value" # 現セルを格納する列名
  ) %>%  # 縦持ちに変換
  ggplot(aes(x = iteration, y = value, color = cluster)) + 
  geom_line() + # 推定値
  geom_hline(yintercept = pi_truth_k, 
             color = "red", linetype = "dashed") + # 真の値
  labs(title = "Maximum Likelihood", 
       subtitle = expression(pi))


# おまけ：アニメーションによる推移を確認 -----------------------------------------------------

# 追加パッケージ
library(gganimate)


# 作図用のデータフレームを作成
trace_model_df <- tibble()
trace_cluster_df <- tibble()
for(i in 1:(MaxIter + 1)) {
  # i回目の推定値による混合分布を計算
  res_dens <- 0
  for(k in 1:K) {
    # クラスタkの分布の確率密度を計算
    tmp_dens <- mvnfast::dmvn(
      X = x_point_mat, mu = trace_mu_ikd[i, k, ], sigma = trace_sigma2_iddk[i, , , k]
    )
    
    # K個の分布を線形結合
    res_dens <- res_dens + trace_pi_ik[i, k] * tmp_dens
  }
  
  # i回目の分布をデータフレームに格納
  res_df <- tibble(
    x_1 = x_point_mat[, 1], 
    x_2 = x_point_mat[, 2], 
    density = res_dens, 
    label = paste0(
      "iter:", i - 1, 
      ", L=", round(trace_L_i[i], 1), 
      ", N=", N, 
      ", pi=(", paste0(round(trace_pi_ik[i, ], 3), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # 観測データとi回目のクラスタ番号をデータフレームに格納
  tmp_gamma_nk <- trace_gamma_ink[i, , ] # i回目の結果
  gamma_df <- tibble(
    x_n1 = x_nd[, 1], 
    x_n2 = x_nd[, 2], 
    cluster = as.factor(max.col(tmp_gamma_nk)), # 負担率が最大のクラスタ番号
    prob = tmp_gamma_nk[cbind(1:N, max.col(tmp_gamma_nk))], # 負担率の最大値
    label = paste0(
      "iter:", i - 1, 
      ", L=", round(trace_L_i[i], 1), 
      ", N=", N, 
      ", pi=(", paste0(round(trace_pi_ik[i, ], 3), collapse = ", "), ")"
    ) %>% 
      as.factor()
  )
  
  # 結果を結合
  trace_model_df <- rbind(trace_model_df, res_df)
  trace_cluster_df <- rbind(trace_cluster_df, gamma_df)
  
  # 動作確認
  print(paste0(i - 1, ' (', round((i - 1) / MaxIter * 100, 1), '%)'))
}


# 分布の推移を作図
trace_model_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour(data = trace_model_df, aes(x = x_1, y = x_2, z = density, color = ..level..)) + # 推定値による分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2)) + # 観測データ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Mixture of Gaussians:Maximum Likelihood", 
       subtitle = paste0("{current_frame}"), 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_model_graph, nframes = MaxIter + 1, fps = 10)


# クラスタの推移を作図
trace_cluster_graph <- ggplot() + 
  geom_contour(data = model_df, aes(x = x_1, y = x_2, z = density), 
               color = "red", alpha = 0.5, linetype = "dashed") + # 真の分布
  geom_point(data = mu_df, aes(x = x_1, y = x_2), 
             color = "red", shape = 4, size = 5) + # 真の平均
  geom_contour_filled(data = trace_model_df, aes(x = x_1, y = x_2, z = density), 
                      alpha = 0.6) + # 推定値による分布
  geom_point(data = trace_cluster_df, aes(x = x_n1, y = x_n2, color = cluster)) + # 負担率によるクラスタ
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Mixture of Gaussians:Maximum Likelihood", 
       subtitle = "{current_frame}", 
       x = expression(x[1]), y = expression(x[2]))

# gif画像を作成
gganimate::animate(trace_cluster_graph, nframes = MaxIter + 1, fps = 10)



