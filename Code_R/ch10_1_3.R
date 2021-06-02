
# 10.1.3 一変数ガウス分布の変分推論 ----------------------------------------------------

# 10.1.3で利用するパッケージ
library(tidyverse)


### 真の分布(1次元ガウス分布)の設定 -----

# 真の平均パラメータを指定
mu_truth <- 5

# 真の精度パラメータを指定
tau_truth <- 0.5
sqrt(1 / tau_truth) # 標準偏差


# 作図用のxの値を作成
x_vec <- seq(
  mu_truth - 4 * sqrt(1 / tau_truth), 
  mu_truth + 4 * sqrt(1 / tau_truth), 
  length.out = 1000
)

# 真の分布を計算
model_df <- tibble(
  x = x_vec, 
  density = dnorm(x = x, mean = mu_truth, sd = sqrt(1 / tau_truth))
)

# 真の分布を作図
ggplot(model_df, aes(x = x, y = density)) + 
  geom_line(color = "blue") + # 真の分布
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("mu=", mu_truth, ", tau=", tau_truth))


### データの生成 -----

# (観測)データ数を指定
N <- 50

# ガウス分布に従うデータを生成
x_n <- rnorm(n = N, mean = mu_truth, sd = sqrt(1 / tau_truth))

# 観測データを確認
summary(x_n)


# 観測データをデータフレームに格納
data_df <- tibble(x_n = x_n)

# 観測データのヒストグラムを作成
ggplot() + 
  #geom_histogram(data = data_df, aes(x = x_n), binwidth = 0.5) + # 観測データ:(度数)
  geom_histogram(data = data_df, aes(x = x_n, y = ..density..), binwidth = 0.5) + # 観測データ:(相対度数)
  geom_line(data = model_df, aes(x = x, y = density), 
            color = "red", linetype = "dashed") + # 真の分布:(相対度数用)
  labs(title = "Gaussian Distribution", 
       subtitle = paste0("N=", N, ", mu=", mu_truth, ", tau=", tau_truth), 
       x = "x")


### 事前分布(ガウス-ガンマ分布)の設定 -----

# muの事前分布のパラメータを指定
mu_0 <- 0
lambda_0 <- 0.1

# tauの事前分布のパラメータを指定
a_0 <- 1
b_0 <- 1


# 作図用のmuの値を作成
mu_vec <- seq(
  mu_truth - 4 * sqrt(1 / tau_truth), 
  mu_truth + 4 * sqrt(1 / tau_truth), 
  length.out = 500
)

# 作図用のtauの値を作成
tau_vec <- seq(0, 4 * tau_truth, length.out = 500)

# 作図用の点を作成
mu_tau_point_mat <- cbind(
  mu = rep(mu_vec, times = length(tau_vec)), 
  tau = rep(tau_vec, each = length(mu_vec))
)


# 真のパラメータをデータフレームに格納
true_params_df <- tibble(mu = mu_truth, tau = tau_truth)


# 同時事前分布を計算
prior_df <- tibble(
  mu = mu_tau_point_mat[, 1], 
  tau = mu_tau_point_mat[, 2], 
  N_dens = dnorm(x = mu, mean = mu_0, sd = sqrt(1 / (lambda_0 * tau + 1e-7))), # muの事前分布
  Gam_dens = dgamma(x = tau, shape = a_0, rate = b_0), # tauの事前分布
  density = N_dens * Gam_dens
)

# 事前分布を作図
ggplot() + 
  geom_contour(data = prior_df, aes(x = mu, y = tau, z = density, color = ..level..)) + # 事前分布
  geom_point(data = true_params_df, aes(x = mu, y = tau), 
             color = "red", shape = 4, size = 5) + # 真の値
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = paste0("mu_0=", mu_0, ", lambda_0=", lambda_0, 
                         ", a_0=", a_0, ", b_0=", b_0), 
       x = expression(mu), y = expression(tau))


### 真の事後分布(ガウス-ガンマ分布)の計算 -----

# muの真の事後分布のパラメータを計算
lambda_hat <- lambda_0 + N
mu_hat <- (lambda_0 * mu_0 + sum(x_n)) / lambda_hat

# lambdaの真の事後分布のパラメータを計算
a_hat <- a_0 + 0.5 * N
b_hat <- b_0 + 0.5 * (sum(x_n^2) + lambda_0 * mu_0^2 - lambda_hat * mu_hat^2)


# 真の同時事後分布を計算
true_posterior_df <- tibble(
  mu = mu_tau_point_mat[, 1], 
  tau = mu_tau_point_mat[, 2], 
  N_dens = dnorm(x = mu, mean = mu_hat, sd = sqrt(1 / (lambda_hat * tau + 1e-7))), # muの真の事後分布
  Gam_dens = dgamma(x = tau, shape = a_hat, rate = b_hat), # tauの真の事後分布
  density = N_dens * Gam_dens
)

# 真の事後分布を作図
ggplot() + 
  geom_contour(data = true_posterior_df, aes(x = mu, y = tau, z = density, color = ..level..)) + # 真の事後分布
  geom_point(data = true_params_df, aes(x = mu, y = tau), 
             color = "red", shape = 4, size = 5) + # 真の値
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = paste0("N=", N, 
                         ", mu_hat=", round(mu_hat, 1), ", lambda_hat=", lambda_hat, 
                         ", a_hat=", a_hat, ", b_hat=", round(b_hat, 1)), 
       x = expression(mu), y = expression(tau))


# muの真の事後分布を作図
tibble(
  mu = mu_vec, 
  N_dens = dnorm(x = mu, mean = mu_hat, sd = sqrt(1 / (lambda_hat * a_hat / b_hat)))
) %>% 
  ggplot(aes(x = mu, y = N_dens)) + 
    geom_line() + # muの真の事後分布
    geom_vline(xintercept = mu_truth, color = "red", linetype = "dashed") + # muの真の値
    labs(title = "Gaussian Distribution", 
         subtitle = paste0("N=", N, 
                           ", mu_hat=", round(mu_hat, 1), 
                           ", lambda_hat=", lambda_hat, 
                           ", E[tau]=", round(a_hat / b_hat, 5)), 
         x = expression(mu), y = "density")

# tauの真の事後分布を作図
tibble(
  tau = tau_vec, 
  Gam_dens = dgamma(x = tau, shape = a_hat, rate = b_hat)
) %>% 
  ggplot(aes(x = tau, y = Gam_dens)) + 
    geom_line() + # tauの真の事後分布
    geom_vline(xintercept = tau_truth, color = "red", linetype = "dashed") + # tauの真の値
    labs(title = "Gamma Distribution", 
         subtitle = paste0("N=", N, ", a_hat=", a_hat, ", b_hat=", round(b_hat, 1)), 
         x = expression(tau), y = "density")


### 推論処理 -----

# 試行回数を指定
MaxIter <- 5

# 初期値を代入
mu_N <- mu_0
lambda_N <- lambda_0 * a_0 / b_0
a_N <- a_0
b_N <- b_0

# 推移の確認用の受け皿を作成
trace_mu_i     <- rep(0, MaxIter * 2 + 1)
trace_lambda_i <- rep(0, MaxIter * 2 + 1)
trace_a_i      <- rep(0, MaxIter * 2 + 1)
trace_b_i      <- rep(0, MaxIter * 2 + 1)

# 初期値を記録
trace_mu_i[1]     <- mu_N
trace_lambda_i[1] <- lambda_N
trace_a_i[1]      <- a_N
trace_b_i[1]      <- b_N

# 変分推論
for(i in 1:MaxIter) {
  
  # muの近似事後分布のパラメータを計算:式(10.26)(10.27)
  mu_N <- (lambda_0 * mu_0 + sum(x_n)) / (lambda_0 + N)
  lambda_N <- (lambda_0 + N) * a_N / b_N
  
  # i回目のmuの近似事後分布の更新後の結果を記録
  trace_mu_i[i * 2]     <- mu_N
  trace_lambda_i[i * 2] <- lambda_N
  trace_a_i[i * 2]      <- a_N
  trace_b_i[i * 2]      <- b_N
  
  
  # tauの近似事後分布のパラメータを計算:式(10.29)(10.30)
  a_N <- a_0 + 0.5 * (N + 1)
  term_x <- 0.5 * (lambda_0 * mu_0^2 + sum(x_n^2))
  term_mu2 <- 0.5 * (lambda_0 + N) * (mu_N^2 + 1 / lambda_N)
  term_mu <- (lambda_0 * mu_0 + sum(x_n)) * mu_N
  b_N <- b_0 + term_x + term_mu2 - term_mu
  
  # i回目のtauの近似事後分布の更新後の結果を記録
  trace_mu_i[i * 2 + 1]     <- mu_N
  trace_lambda_i[i * 2 + 1] <- lambda_N
  trace_a_i[i * 2 + 1]      <- a_N
  trace_b_i[i * 2 + 1]      <- b_N
  
  # 動作確認
  #print(paste0(i, ' (', round(i / MaxIter * 100, 1), '%)'))
}


### 推論結果の確認 -----

# 同時近似事後分布を計算
E_tau <- a_N / b_N # tauの期待値
posterior_df <- tibble(
  mu = mu_tau_point_mat[, 1], 
  tau = mu_tau_point_mat[, 2], 
  N_dens = dnorm(x = mu, mean = mu_N, sd = sqrt(1 / (lambda_N / E_tau * tau + 1e-7))), # muの近似事後分布
  Gam_dens = dgamma(x = tau, shape = a_N, rate = b_N), # tauの近似事後分布
  density = N_dens * Gam_dens
)

# 近似事後分布を作図
ggplot() + 
  geom_contour(data = posterior_df, aes(x = mu, y = tau, z = density, color = ..level..)) + # 近似事後分布
  geom_contour(data = true_posterior_df, aes(x = mu, y = tau, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の事後分布
  geom_point(data = true_params_df, aes(x = mu, y = tau), 
             color = "red", shape = 4, size = 5) + # 真の値
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = paste0("N=", N, 
                         ", mu_N=", round(mu_N, 1), 
                         ", lambda_N / E[tau]=", round(lambda_N / E_tau, 1), 
                         ", a_N=", a_N, 
                         ", b_N=", round(b_N, 1)), 
       x = expression(mu), y = expression(tau))


# muの近似事後分布を作図
tibble(
  mu = mu_vec, 
  N_dens = dnorm(x = mu, mean = mu_N, sd = sqrt(1 / lambda_N))
) %>% 
  ggplot(aes(x = mu, y = N_dens)) + 
    geom_line() + # muの近似事後分布
    geom_vline(xintercept = mu_truth, color = "red", linetype = "dashed") + # muの真の値
    labs(title = "Gaussian Distribution", 
         subtitle = paste0("N=", N, 
                           ", mu_N=", round(mu_N, 1), 
                           ", lambda_N=", round(lambda_N, 5)), 
        x = expression(mu), y = "density")

# tauの近似事後分布を作図
tibble(
  tau = tau_vec, 
  Gam_dens = dgamma(x = tau, shape = a_N, rate = b_N)
) %>% 
  ggplot(aes(x = tau, y = Gam_dens)) + 
    geom_line() + # tauの近似事後分布
    geom_vline(xintercept = tau_truth, color = "red", linetype = "dashed") + # tauの真の値
    labs(title = "Gamma Distribution", 
         subtitle = paste0("N=", N, ", a_N=", a_N, ", b_N=", round(b_N, 1)), 
         x = expression(tau), y = "density")


# 値を確認
mu_hat; mu_N
lambda_hat; lambda_N * b_N / a_N
lambda_hat * a_hat / b_hat; lambda_N
a_hat; a_N
b_hat; b_N
a_hat / b_hat; a_N / b_N


### 超パラメータの推移の確認 -----

# mu_Nの推移を作図
tibble(
  iteration = 0:(MaxIter * 2) * 0.5, 
  value = trace_mu_i
) %>% 
  ggplot(aes(x = iteration, y = value)) + 
    geom_line() + 
    labs(title = "Variational Inference", 
         subtitle = expression(mu[N]))

# lambda_Nの推移を作図
tibble(
  iteration = 0:(MaxIter * 2) * 0.5, 
  value = trace_lambda_i
) %>% 
  ggplot(aes(x = iteration, y = value)) + 
    geom_line() + 
    labs(title = "Variational Inference", 
         subtitle = expression(lambda[N]))

# a_Nの推移を作図
tibble(
  iteration = 0:(MaxIter * 2) * 0.5, 
  value = trace_a_i
) %>% 
  ggplot(aes(x = iteration, y = value)) + 
    geom_line() + 
    labs(title = "Variational Inference", 
         subtitle = expression(a[N]))

# b_Nの推移を作図
tibble(
  iteration = 0:(MaxIter * 2) * 0.5, 
  value = trace_b_i
) %>% 
  ggplot(aes(x = iteration, y = value)) + 
    geom_line() + 
    labs(title = "Variational Inference", 
         subtitle = expression(b[N]))


# ・アニメーションによる推移の確認 --------------------------------------------------------

# 追加パッケージ
library(gganimate)


# 作図用のデータフレームを作成
trace_posterior_df <- tibble()
for(i in 1:(MaxIter * 2 + 1)) {
  
  # i回目の近似事後分布を計算
  E_tau <- trace_a_i[i] / trace_b_i[i] # tauの期待値
  tmp_posterior_df <- tibble(
    mu = mu_tau_point_mat[, 1], 
    tau = mu_tau_point_mat[, 2], 
    N_dens = dnorm(
      x = mu, 
      mean = trace_mu_i[i], 
      sd = sqrt(1 / (trace_lambda_i[i] / E_tau * tau + 1e-7))
    ), # muの近似事後分布
    Gam_dens = dgamma(x = tau, shape = trace_a_i[i], rate = trace_b_i[i]), # tauの近似事後分布
    density = N_dens * Gam_dens, 
    label = as.factor(
      paste0("iter:", (i - 1) * 0.5, ", N=", N, 
             ", mu_N=", round(trace_mu_i[i], 1), ", lambda_N=", round(trace_lambda_i[i], 5), 
             ", a_N=", trace_a_i[i], ", b_N=", round(trace_b_i[i], 1))
    ) # フレーム切替用のラベル
  )
  
  # 結合
  trace_posterior_df <- rbind(trace_posterior_df, tmp_posterior_df)
  
  # 動作確認
  print(paste0((i - 1) * 0.5, ' (', round((i - 1) * 0.5 / MaxIter * 100, 1), '%)'))
}


# 近似事後分布を作図
trace_graph <- ggplot() + 
  geom_contour(data = trace_posterior_df, aes(x = mu, y = tau, z = density, color = ..level..)) + # 近似事後分布
  geom_contour(data = true_posterior_df, aes(x = mu, y = tau, z = density, color = ..level..), 
               alpha = 0.5, linetype = "dashed") + # 真の事後分布
  geom_point(data = true_params_df, aes(x = mu, y = tau), color = "red", shape = 4, size = 5) + # 真の値
  gganimate::transition_manual(label) + # フレーム
  labs(title = "Gaussian-Gamma Distribution", 
       subtitle = "{current_frame}", 
       x = expression(mu), y = expression(tau))


# gif画像を作成
gganimate::animate(trace_graph, nframes = MaxIter * 2 + 1, fps = 5)


