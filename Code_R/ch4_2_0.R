
# ch4.2.0 ロジスティックシグモイド関数 ----------------------------------------------------------

# 4.2.0項で利用するパッケージ
library(tidyverse)


# ・ロジスティックシグモイド関数 ----------------------------------------------------------

# ロジスティックシグモイド関数を計算
sigmoid_df <- tidyr::tibble(
  a = seq(-10, 10, by = 0.05), # x軸の範囲を指定
  sigma = 1 / (1 + exp(-a))
)

# ロジスティックシグモイド関数を作図
ggplot(sigmoid_df, aes(x = a, y = sigma)) + 
  geom_line() + 
  labs(title = "Logistic Sigmoid Function", 
       y = expression(sigma))


### 対称性 -----

# シグモイド関数を作成
sigmoid <- function(a) {
  y <- 1 / (1 + exp(-a))
  return(y)
}

# 入力の値を指定
a <- 2

# 対称性をグラフで確認
ggplot(sigmoid_df, aes(x = a, y = sigma)) + 
  geom_line() + # シグモイド関数
  geom_vline(xintercept = -a, alpha = 0.5, linetype = "dashed") + # x = -aの補助線
  geom_vline(xintercept = a, alpha = 0.5, linetype = "dashed") + # x = aの補助線
  geom_linerange(x = -a, ymin = sigmoid(-a), ymax = 1, color = "red") + # 1 - σ(-a)の線分
  geom_linerange(x = a, ymin = 0, ymax = sigmoid(a), color = "red") + # σ(a)の線分
  scale_x_continuous(breaks = c(-a, a), labels = c("-a", "a")) + # x軸ラベル
  labs(title = "Logistic Sigmoid Function", 
       subtitle = paste0("a=", a), 
       x = "x", y = "y")

# 対称性をグラフで確認
ggplot(sigmoid_df, aes(x = a, y = sigma)) + 
  geom_line() + # シグモイド関数
  geom_vline(xintercept = -a, alpha = 0.5, linetype = "dashed") + # x = -aの補助線
  geom_vline(xintercept = a, alpha = 0.5, linetype = "dashed") + # x = aの補助線
  geom_linerange(x = -a, ymin = 0, ymax = sigmoid(-a), color = "red") + # σ(-a)の線分
  geom_linerange(x = a, ymin = sigmoid(a), ymax = 1, color = "red") + # 1 - σ(a)の線分
  scale_x_continuous(breaks = c(-a, a), labels = c("-a", "a")) + # x軸ラベル
  labs(title = "Logistic Sigmoid Function", 
       subtitle = paste0("a=", a), 
       x = "x", y = "y")


### 微分 -----

# ロジスティックシグモイド関数を計算
sigmoid_df <- tidyr::tibble(
  x = seq(-10, 10, by = 0.05), # 入力の範囲を指定
  y = 1 / (1 + exp(-x)), # 出力
  dy = y * (1 - y) # 微分
) %>% 
  tidyr::pivot_longer(cols = !x, names_to = "type", values_to = "value") #  縦長のデータフレームに変換

# ロジスティックシグモイド関数を作図
ggplot(sigmoid_df, aes(x = x, y = value, color = type)) + 
  geom_line() + 
  labs(title = "Logistic Sigmoid Function")


# ・ロジット関数 -----------------------------------------------------------------

# ロジット関数を計算
logit_df <- tidyr::tibble(
  sigma = seq(0, 1, by = 0.01), # 入力の範囲
  a = log(sigma / (1 - sigma))
)

# ロジット関数を作図
ggplot(logit_df, aes(x = sigma, y = a)) + 
  geom_line() + 
  labs(title = "Logit Function", 
       x = expression(sigma))


