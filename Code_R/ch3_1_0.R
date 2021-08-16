
# 3.1 線形基底関数モデル -----------------------------------------------------------

# 3.1.0項で利用するパッケージ
library(tidyverse)


# 多項式基底関数 ---------------------------------------------------------------------


# 多項式基底関数を作成
phi <- function(x, j) {
  y <- x^j
  return(y)
}

# x軸の値を作成
x_vals <- seq(-1, 1, by = 0.001)


### 基本のグラフ -----

# 値を指定
j <- 3

# 出力を計算
res_df <- tibble::tibble(
  x = x_vals, 
  y = phi(x, j), 
  label = factor(paste0("j=", j))
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Polynomial Basis Function", color = "params", 
       y = expression(phi(x)))


### jの影響 -----

# 値を指定
j_vec <- seq(0, 10)

# 出力を計算
res_df <- tibble::tibble(
  j = rep(j_vec, each = length(x_vals)), 
  x = rep(x_vals, times = length(j_vec)), 
  y = phi(x, j), 
  label = factor(paste0("j=", j), levels = paste0("j=", j_vec))
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Polynomial Basis Function", color = "params", 
       y = expression(phi(x)))


# ガウス基底関数 -----------------------------------------------------------------


# ガウス基底関数を作成
phi <- function(x, mu, s) {
  y <- exp(-(x - mu)^2 / (2 * s^2))
  return(y)
}

# x軸の値を作成
x_vals <- seq(-1, 1, by = 0.001)


### 基本のグラフ -----

# 値を指定
mu <- 0
s <- 0.2

# 出力を計算
res_df <- tibble::tibble(
  x = x_vals, 
  y = phi(x, mu, s), 
  label = factor(paste0("mu=", mu, ", s=", s))
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Gaussian Basis Function", color = "params", 
       y = expression(phi(x)))


### muの影響 -----

# 値を指定
mu_vec <- seq(-1, 1, by = 0.2)
s <- 0.2

# 出力を計算
res_df <- tibble::tibble(
  mu = rep(mu_vec, each = length(x_vals)), 
  x = rep(x_vals, times = length(mu_vec)), 
  y = phi(x, mu, s), 
  label = factor(
    paste0("mu=", mu, ", s=", s), 
    levels = paste0("mu=", mu_vec, ", s=", s)
  )
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Gaussian Basis Function", color = "params", 
       y = expression(phi(x)))


### sの影響 -----

# 値を指定
mu <- 0
s_vec <- seq(0.1, 1, by = 0.1)

# 出力を計算
res_df <- tibble::tibble(
  s = rep(s_vec, each = length(x_vals)), 
  x = rep(x_vals, times = length(s_vec)), 
  y = phi(x, mu, s), 
  label = factor(
    paste0("mu=", mu, ", s=", s), 
    levels = paste0("mu=", mu, ", s=", s_vec)
  )
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Gaussian Basis Function", color = "params", 
       y = expression(phi(x)))


# シグモイド基底関数 ----------------------------------------------------------


# ロジスティックシグモイド関数を作成
sigma <- function(x) {
  y = 1 / (1 + exp(-x))
}

# シグモイド基底関数を作成
phi <- function(x, mu, s) {
  a <- (x - mu) / s
  y <- sigma(a)
  return(y)
}

# x軸の値を作成
x_vals <- seq(-1, 1, by = 0.001)


### 基本のグラフ -----

# 値を指定
mu <- 0
s <- 0.1

# 出力を計算
res_df <- tibble::tibble(
  x = x_vals, 
  y = phi(x, mu, s), 
  label = factor(paste0("mu=", mu, ", s=", s))
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Sigmoid Basis Function", color = "params", 
       y = expression(phi(x)))


### muの影響 -----

# 値を指定
mu_vec <- seq(-1, 1, by = 0.2)
s <- 0.1

# 出力を計算
res_df <- tibble::tibble(
  mu = rep(mu_vec, each = length(x_vals)), 
  x = rep(x_vals, times = length(mu_vec)), 
  y = phi(x, mu, s), 
  label = factor(
    paste0("mu=", mu, ", s=", s), 
    levels = paste0("mu=", mu_vec, ", s=", s)
  )
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Sigmoid Basis Function", color = "params", 
       y = expression(phi(x)))


### sの影響 -----

# 値を指定
mu <- 0
s_vec <- seq(0.1, 1, by = 0.1)

# 出力を計算
res_df <- tibble::tibble(
  s = rep(s_vec, each = length(x_vals)), 
  x = rep(x_vals, times = length(s_vec)), 
  y = phi(x, mu, s), 
  label = factor(
    paste0("mu=", mu, ", s=", s), 
    levels = paste0("mu=", mu, ", s=", s_vec)
  )
)

# 作図
ggplot(res_df, aes(x = x, y = y, color = label)) + 
  geom_line() + 
  labs(title = "Sigmoid Basis Function", color = "params", 
       y = expression(phi(x)))


