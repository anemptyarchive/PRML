
# ch3.1.4：ラッソ回帰 ----------------------------------------------------------

# 3.1.4項で利用するパッケージ
library(tidyverse)
library(gganimate)


# 共通の関数 -------------------------------------------------------------------

# ラッソ回帰の重みパラメータの最尤解を計算する関数を作成
soft_thresholding <- function(S, lambda, phi_x_n) {
  # 条件に応じて値を計算
  if(S > lambda) {
    # S > λの場合の計算
    w <- (S - lambda) / sum(phi_x_n^2)
  } else if(S < -lambda) {
    # S < -λの場合の計算
    w <- (S + lambda) / sum(phi_x_n^2)
  } else {
    # -λ =< S =< λの場合
    w <- 0
  }
  return(w)
}


# ・ラッソ回帰 -------------------------------------------------------------------

### モデルの設定とデータの生成 -----

# ch3_1_1.Rを参照


### 最尤推定 -----

# 繰り返し回数を指定
max_iter <- 100

# 正則化係数を指定
lambda <- 0.5

# 重みパラメータを初期化
w_lasso_m <- runif(n = M, min = -5, max = 5) # 範囲を指定

# 座標降下法による推定
for(i in 1:max_iter) {
  # バイアスパラメータを0に置換:(バイアスを正則化項に含めない場合の処理)
  w_lasso_m[1] <- 0
  
  # バイアスパラメータの最尤解を計算:(バイアスを正則化項に含めない場合の処理)
  w_lasso_m[1] <- sum(t_n - phi_x_nm %*% w_lasso_m) / N
  
  # パラメータを要素ごとに更新
  for(m in 2:M) { # (バイアスを正則化項に含める場合は1からMに変更)
    # m番目のパラメータを0に置換
    w_lasso_m[m] <- 0
    
    # 分子の項を計算
    S <- sum((t_n - phi_x_nm %*% w_lasso_m) * phi_x_nm[, m])
    
    # 重みパラメータの最尤解を計算
    w_lasso_m[m] <- soft_thresholding(S, lambda, phi_x_nm[, m])
  }
}

# ラッソ回帰の分散パラメータの最尤解を計算
sigma2_lasso <- sum((t_n - phi_x_nm %*% w_lasso_m)^2) / N

# リッジ回帰の精度パラメータの最尤解を計算
beta_lasso <- 1 / sigma2_lasso


# 推定したパラメータによるモデルを計算
lasso_df <- tidyr::tibble(
  x = x_vals, # x軸の値
  t = Phi(x_vals, M) %*% w_lasso_m %>% 
    as.vector(), # y軸の値
  minus_sigma = t - 2 * sqrt(sigma2_lasso), # μ - 2σ
  plus_sigma = t + 2 * sqrt(sigma2_lasso) # μ + 2σ
)

# 推定したパラメータによるモデルを作図
ggplot() + 
  geom_point(data = data_df, aes(x = x_n, y = t_n)) + # 観測データ
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_ribbon(data = model_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "cyan4", alpha = 0.1, color = "cyan4", linetype = "dotted") + # 真のノイズ範囲
  geom_line(data = ml_df, aes(x = x, y = t), color = "blue", linetype ="dashed") + # 推定したモデル:(正則化なし)
  geom_line(data = lasso_df, aes(x = x, y = t), color = "blue") + # 推定したモデル:(L1正則化)
  geom_ribbon(data = lasso_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "blue", alpha = 0.1, color = "blue", linetype = "dotted") + # 推定したノイズ範囲
  #ylim(c(-5, 5)) + # y軸の表示範囲
  labs(title = "Lasso Regression", 
       subtitle = paste0("N=", N, ", M=", M, ", q=", 1, ", lambda=", lambda, 
                         ", w=(", paste0(round(w_lasso_m, 2), collapse = ", "), ")", 
                         ", beta=", round(beta_lasso, 2)), 
       x = "x", y = "t")


### 正則化係数と回帰曲線の関係をアニメーションで確認 -----

# 使用するlambdaの値を作成
lambda_vals <- seq(0, 1, by = 0.01)
length(lambda_vals) # フレーム数

# 重みパラメータを初期値を生成
w_init_m <- runif(n = M, min = -5, max = 5) # 範囲を指定

# lambdaごとに最尤解を計算
anime_lasso_df <- dplyr::tibble()
for(lambda in lambda_vals) {
  # 重みパラメータを初期化
  w_lasso_m <- w_init_m
  
  # 座標降下法による推定
  for(i in 1:max_iter) {
    # バイアスパラメータを0に置換:(バイアスを正則化項に含めない場合の処理)
    w_lasso_m[1] <- 0
    
    # バイアスパラメータの最尤解を計算:(バイアスを正則化項に含めない場合の処理)
    w_lasso_m[1] <- sum(t_n - phi_x_nm %*% w_lasso_m) / N
    
    # パラメータを要素ごとに更新
    for(m in 2:M) { # (バイアスを正則化項に含める場合は1からMに変更)
      # m番目のパラメータを0に置換
      w_lasso_m[m] <- 0
      
      # 分子の項を計算
      S <- sum((t_n - phi_x_nm %*% w_lasso_m) * phi_x_nm[, m])
      
      # 重みパラメータの最尤解を計算
      w_lasso_m[m] <- soft_thresholding(S, lambda, phi_x_nm[, m])
    }
  }
  
  # ラッソ回帰の分散パラメータの最尤解を計算
  sigma2_lasso <- sum((t_n - phi_x_nm %*% w_lasso_m)^2) / N
  
  # ラッソ回帰の精度パラメータの最尤解を計算
  beta_lasso <- 1 / sigma2_lasso
  
  # 推定したパラメータによるモデルを計算
  tmp_lasso_df <- tidyr::tibble(
    x = x_vals, # x軸の値
    t = Phi(x_vals, M) %*% w_lasso_m %>% 
      as.vector(), # y軸の値
    minus_sigma = t - 2 * sqrt(sigma2_lasso), # μ - 2σ
    plus_sigma = t + 2 * sqrt(sigma2_lasso), # μ + 2σ
    label = paste0(
      "lambda=", lambda, 
      ", w=(", paste0(round(w_lasso_m, 2), collapse = ", "), ")", 
      ", beta=", round(beta_lasso, 2)
    ) %>% 
      as.factor() # フレーム切替用のラベル
  )
  
  # 結果を結合
  anime_lasso_df <- rbind(anime_lasso_df, tmp_lasso_df)
}

# 推定したパラメータによるモデルを作図
anime_graph <- ggplot() + 
  geom_point(data = data_df, aes(x = x_n, y = t_n)) + # 観測データ
  geom_line(data = model_df, aes(x = x, y = y), color = "cyan4") + # 真のモデル
  geom_ribbon(data = model_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "cyan4", alpha = 0.1, color = "cyan4", linetype = "dotted") + # 真のノイズ範囲
  geom_line(data = ml_df, aes(x = x, y = t), color = "blue", linetype ="dashed") + # 推定したモデル:(正則化なし)
  geom_line(data = anime_lasso_df, aes(x = x, y = t), color = "blue") + # 推定したモデル:(L1正則化)
  geom_ribbon(data = anime_lasso_df, aes(x = x, ymin = minus_sigma, ymax = plus_sigma), 
              fill = "blue", alpha = 0.1, color = "blue", linetype = "dotted") + # 推定したノイズ範囲
  gganimate::transition_manual(label) + # フレーム
  #ylim(c(-3, 3)) + # y軸の表示範囲
  labs(title = "Lasso Regression", 
       subtitle = "{current_frame}", 
       x = "x", y = "t")

# gif画像に変換
gganimate::animate(anime_graph, nframes = length(lambda_vals), fps = 10)


### 正則化係数と劣微分の関係 -----

# 正則化係数との大小関係によりパラメータを計算
subdiff_df <- tidyr::tibble(
  S = seq(-5, 5, by = 0.1), # 分子の項
  w = dplyr::case_when(
    S > lambda ~ (S - lambda) / sum(phi_x_nm[, m]^2), 
    S < -lambda ~ (S + lambda) / sum(phi_x_nm[, m]^2), 
    TRUE ~ 0
  ) # 重みパラメータ
)

# 正則化係数との大小関係によるパラメータのグラフを作成
ggplot(subdiff_df, aes(x = S, y = w)) + 
  geom_line() + # 重みパラメータ
  labs(title = expression(paste(w[k] == 0, " (", - lambda <= S, ", ", S <= lambda, ")")), 
       subtitle = paste0("lambda=", lambda), 
       y = expression(w[k]))



# ・重みパラメータと誤差関数の関係 -----------------------------------------------------------

### モデルの設定 -----

# 真の関数を指定
y_true <- function(x) {
  # 計算式を指定
  y <- sin(pi * x)
  return(y)
}

# 基底関数を指定
Phi <- function(x) {
  # 計算式を指定
  phi_x <- cbind(x, x^2)
  return(phi_x)
}


# データ数を指定
N <- 50

# (観測)データを生成
x_n <- runif(n = N, min = 0, max = 1) # 入力
t_n <- y_true(x_n) + rnorm(n = N, mean = 0, sd = 1) # 出力

# 基底関数により入力を変換
phi_x_nm <- Phi(x_n)


# 値を設定:(固定)
q <- 1

# パラメータの次元数を設定:(固定)
M <- 2

# 作図用のwの範囲を指定
w_i <- seq(-5, 5, by = 0.1)

# 作図用のwの点を作成
w_im <- expand.grid(w_i, w_i) %>% 
  as.matrix()

# 正則化項を計算
E_df <- tidyr::tibble(
  w_1 = w_im[, 1], # x軸の値
  w_2 = w_im[, 2], # y軸の値
  E_D = colSums((t_n - phi_x_nm %*% t(w_im))^2) / N, # 二乗和誤差
  E_W = abs(w_1)^q + abs(w_2)^q # 正則化項
)

# 二乗和誤差関数の等高線図を作成
ggplot(E_df, aes(x = w_1, y = w_2)) + 
  geom_contour_filled(aes(z = E_D, fill = ..level..), alpha = 0.7) + # 二乗和誤差関数:(塗りつぶし)
  #geom_contour(aes(z = E_D, color = ..level..)) + # 二乗和誤差関数:(等高線)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E[D](w)), 
       subtitle = paste0("q=", q), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[D](w)))

# 正則化項の等高線図を作成
ggplot(E_df, aes(x = w_1, y = w_2)) + 
  geom_contour_filled(aes(z = E_W, fill = ..level..), alpha = 0.7) + # 正則化項:(塗りつぶし)
  #geom_contour(aes(z = E_W, color = ..level..)) + # 正則化項:(等高線)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E[W](w)), 
       subtitle = paste0("q=", q), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[W](w)))


### 最尤推定 -----

# 繰り返し回数を指定
max_iter <- 50

# 正則化係数を指定
lambda <- 2.5

# 重みパラメータを初期化
w_lasso_m <- runif(n = M, min = -5, max = 5)

# 座標降下法による推定
for(i in 1:max_iter) {
  
  # パラメータを要素ごとに更新
  for(m in 1:M) {
    # m番目のパラメータを0に置換
    w_lasso_m[m] <- 0
    
    # 分子の項を計算
    S <- sum((t_n - phi_x_nm %*% w_lasso_m) * phi_x_nm[, m])
    
    # ラッソ回帰の重みパラメータの最尤解を計算
    w_lasso_m[m] <- soft_thresholding(S, lambda, phi_x_nm[, m])
  }
}

# 重みパラメータの最尤解を計算
w_ml_m <- solve(t(phi_x_nm) %*% phi_x_nm) %*% t(phi_x_nm) %*% t_n %>% 
  as.vector()

# 重みパラメータを格納
w_df <- tidyr::tibble(
  w_1 = c(w_lasso_m[1], w_ml_m[1]), # x軸の値
  w_2 = c(w_lasso_m[2], w_ml_m[2]), # y軸の値
  method = factor(c("lasso", "ml"), levels = c("lasso", "ml")) # ラベル
)

# 推定したパラメータによる誤差項を計算
E_D_val <- sum((t_n - phi_x_nm %*% w_lasso_m)^2) / N
E_W_val <- sum(abs(w_lasso_m)^q)

# 最尤解を作図
ggplot() + 
  geom_contour_filled(data = E_df, aes(x = w_1, y = w_2, z = E_D, fill = ..level..), alpha = 0.7) + # 二乗和誤差関数:(塗りつぶし)
  geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_D), color = "blue", breaks = E_D_val) + # 二乗和誤差関数:(等高線)
  geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_W), color = "red", breaks = E_W_val) + # 正則化項:(等高線)
  geom_point(data = w_df, aes(x = w_1, y = w_2, color = method), shape = 4, size = 5) + # パラメータの最尤解
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Lasso Regression", 
       subtitle = paste0("lambda=", lambda, 
                         ", w_lasso=(", paste0(round(w_lasso_m, 2), collapse = ", "), ")", 
                         ", w_ml=(", paste0(round(w_ml_m, 2), collapse = ", "), ")"), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[D](w)))


# 誤差関数を作図
ggplot(E_df, aes(x = w_1, y = w_2)) + 
  geom_contour_filled(aes(z = E_D+lambda*E_W, fill = ..level..)) + # 二乗和誤差関数:(塗りつぶし)
  geom_contour(aes(z = E_D), color = "blue", alpha = 0.7, linetype = "dashed") + # 二乗和誤差関数:(等高線)
  geom_contour(aes(z = E_W), color = "red", alpha = 0.7, linetype = "dashed") + # 正則化項:(等高線)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E(w) == E[D](w) + lambda * E[w](w)), 
       subtitle = paste0("q=", q, ", lambda=", lambda), 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E(w)))


### 正則化係数と最尤解の関係をアニメーションで確認 ----

# 接線用に細かいwの点を作成
w_vals <- seq(-5, 5, by = 0.005) # 刻み幅を変更
w_point <- expand.grid(w_vals, w_vals) %>% 
  as.matrix()
hd_E_df <- tidyr::tibble(
  w_1 = w_point[, 1], # x軸の値
  w_2 = w_point[, 2], # y軸の値
  E_D = colSums((t_n - phi_x_nm %*% t(w_point))^2) / N, # 二乗和誤差
  E_W = abs(w_1)^q + abs(w_2)^q # 正則化項
)


# 使用するlambdaの値を作成
lambda_vals <- seq(0, 25, by = 0.1)
length(lambda_vals) # フレーム数

# 重みパラメータを初期値を生成
w_init_m <- runif(n = M, min = -5, max = 5)

# lambdaごとに最尤解を計算
anime_w_df <- tidyr::tibble() # パラメータの最尤解
anime_E_D_df <- tidyr::tibble() # 誤差項:(接線)
anime_E_W_df <- tidyr::tibble() # 正則化項:(接線)
anime_E_df <- tidyr::tibble() # 誤差関数
for(lambda in lambda_vals) {
  # 重みパラメータを初期化
  w_lasso_m <- w_init_m
  
  # 座標降下法による推定
  for(i in 1:max_iter) {
    
    # m番目パラメータを更新
    for(m in 1:M) {
      # m番目のパラメータを0に置換
      w_lasso_m[m] <- 0
      
      # 分子の項を計算
      S <- sum((t_n - phi_x_nm %*% w_lasso_m) * phi_x_nm[, m])
      
      # 重みパラメータの最尤解を計算
      w_lasso_m[m] <- soft_thresholding(S, lambda, phi_x_nm[, m])
    }
  }
  
  # 推定したパラメータによる誤差項を計算
  E_D_val <- sum((t_n - phi_x_nm %*% w_lasso_m)^2) / N # 二乗和誤差
  E_W_val <- sum(abs(w_lasso_m)^q) # 正則化項
  
  # アニメーション用のラベルを作成
  label_txt <- paste0(
    "lambda=", lambda, ", E=", round(E_D_val + lambda * E_W_val, 2), 
    ", E_D=", round(E_D_val, 2), ", E_W=", round(E_W_val, 2), 
    ", w=(", paste0(round(w_lasso_m, 2), collapse = ", "), ")"
  )
  
  # 推定したパラメータを格納
  tmp_w_df <- tidyr::tibble(
    w_1 = c(w_lasso_m[1], w_ml_m[1]), # x軸の値
    w_2 = c(w_lasso_m[2], w_ml_m[2]), # y軸の値
    method = factor(c("lasso", "ml"), levels = c(c("lasso", "ml"))), # ラベル
    label = as.factor(label_txt) # フレーム切替用のラベル
  )
  
  # 結果を結合
  anime_w_df <- rbind(anime_w_df, tmp_w_df)
  
  # 接線となる誤差項の等高線を抽出
  anime_E_D_df <- hd_E_df %>% 
    dplyr::select(w_1, w_2, E_D) %>% # 利用する列を抽出
    dplyr::mutate(
      E_D = dplyr::if_else(
        round(E_D, 2) == round(E_D_val, 2), true = round(E_D_val, 2), false = 0
      )
    ) %>% # 接線となる誤差項の点以外を0に置換
    cbind(label = as.factor(label_txt)) %>% # フレーム切替用のラベル列を追加
    rbind(anime_E_D_df, .) %>%  # 結果を結合
    dplyr::filter(E_D > 0) # 接線となる誤差項の点を抽出:(最後でないと接線となる点がなかった時にエラーになる)
  
  # 接線となる正則化項の等高線を抽出
  anime_E_W_df <- hd_E_df %>% 
    dplyr::select(w_1, w_2, E_W) %>% # 利用する列を抽出
    dplyr::mutate(
      E_W = dplyr::if_else(
        round(E_W, 2) == round(E_W_val, 2), true = round(E_W_val, 2), false = 0
      )
    ) %>% # 接線となる正則化項の点以外を0に置換
    cbind(label = as.factor(label_txt)) %>% # フレーム切替用のラベル列を追加
    rbind(anime_E_W_df, .) %>% # 結合
    dplyr::filter(E_W > 0) # 接線となる正則化項の点を抽出:(最後でないと接線となる点がなかった時にエラーになる)
  
  # アニメーション用に複製
  anime_E_df <- E_df %>% 
    dplyr::mutate(E = E_D + lambda * E_W) %>% # 誤差を計算
    #dplyr::select(w_1, w_2, E) %>% # 使用する列を抽出
    cbind(label = as.factor(label_txt)) %>% # フレーム切替用のラベル列を追加
    rbind(anime_E_df, .) # 結果を結合
  
  # 途中経過を表示
  message("\r", rep(" ", 30), appendLF = FALSE) # 前回のメッセージを初期化
  message("\r", "lambda=", lambda, " (", round(lambda / max(lambda_vals) * 100, 2), "%)", appendLF = FALSE)
}


# 誤差項と最尤解の関係を作図
anime_graph <- ggplot() + 
  #geom_contour_filled(data = anime_E_df, aes(x = w_1, y = w_2, z = E, fill = ..level..), alpha = 0.7) + # 誤差関数:(塗りつぶし等高線)
  geom_contour_filled(data = anime_E_df, aes(x = w_1, y = w_2, z = E_D, fill = ..level..), alpha = 0.7) + # 誤差項:(塗りつぶし等高線)
  #geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_D), 
  #             color = "blue", linetype = "dashed", breaks = seq(1, 10, length.out = 3)) + # 誤差項:(等高線)
  geom_contour(data = E_df, aes(x = w_1, y = w_2, z = E_W), 
               color = "red", linetype = "dashed", breaks = seq(1, 3, by = 1)) + # 正則化項:(等高線)
  geom_point(data = anime_E_D_df, aes(x = w_1, y = w_2), color = "blue", shape = ".", size = 0.1) + # 誤差項:(接線)
  geom_point(data = anime_E_W_df, aes(x = w_1, y = w_2), color = "red", shape = ".", size = 0.1) + # 正則化項:(接線)
  geom_point(data = anime_w_df, aes(x = w_1, y = w_2, color = method), shape = 4, size = 5) + # パラメータの最尤解
  gganimate::transition_manual(label) + # フレーム
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Lasso Regression", 
       subtitle = "{current_frame}", 
       x = expression(w[1]), y = expression(w[2]), fill = expression(E[D](w)))

# gif画像に変換
gganimate::animate(anime_graph, nframes = length(lambda_vals), fps = 10)


