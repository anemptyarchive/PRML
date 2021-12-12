
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



# 導関数と接線の関係 ---------------------------------------------------------------

# 追加パッケージ
library(gganimate)


# 関数を指定
f <- function(x) {
  y <- 1 / (1 + exp(-x)) # Sigmoid
  #y <- (exp(x) - exp(-x)) / (exp(x) + exp(-x)) # tanh
  return(y)
}

# 導関数を指定
df <- function(y) { 
  dy <- y * (1 - y) # Sigmoid
  #dy <- 1 - y^2 # tanh
  return(dy)
}


# x軸の値を作成
x_vals <- seq(-10, 10, by = 0.1)
length(x_vals) # フレーム数  

# Sigmoid関数を計算
sigmoid_df <- tidyr::tibble(
  x = x_vals, 
  y = f(x) # 出力
)

# xの値ごとにグラフを計算
anime_diff_df <- tidyr::tibble()
anime_tangentline_df <- tidyr::tibble()
anime_tangentpoint_df <- tidyr::tibble()
for(i in 1:length(x_vals)) {
  # 接点のx軸の値を取得
  x_val <- x_vals[i]
  
  # 接点のy軸の値を計算
  y_val <- f(x_val)
  
  # 接線の傾きを計算
  dy_val <- df(y_val)
  
  # 接線の切片を計算
  b_val <- y_val- dy_val * x_val
  
  # フレーム切替用のラベルを作成
  label_txt <- paste(
    "(x, y)=(", round(x_val, 2), ", ", round(y_val, 2), "), dy=", round(dy_val, 3)
  )
  
  # 導関数を計算
  tmp_diff_df <- tidyr::tibble(
    x = x_vals[1:i], 
    y = f(x), # 出力
    dy = df(y), # 微分
    label = as.factor(label_txt)
  )
  anime_diff_df <- rbind(anime_diff_df, tmp_diff_df)
  
  # 接線を計算
  tmp_tangentline_df <- tidyr::tibble(
    x = x_vals, 
    y = dy_val * x + b_val, 
    label = as.factor(label_txt)
  )
  anime_tangentline_df <- rbind(anime_tangentline_df, tmp_tangentline_df)
  
  # 接点を格納
  tmp_tangentpoint_df <- tidyr::tibble(
    x = x_val, 
    y = y_val, 
    label = as.factor(label_txt)
  )
  anime_tangentpoint_df <- rbind(anime_tangentpoint_df, tmp_tangentpoint_df)
  
  # 途中経過を表示
  message("\r", rep(" ", 20), appendLF = FALSE)
  message("\r", i, " (", round(i / length(x_vals) * 100, 2), "%)", appendLF = FALSE)
}

# 作図
anime_graph <- ggplot() + 
  geom_line(data = sigmoid_df, aes(x = x, y = y, group = 1), color = "blue") + # 対象の関数
  geom_line(data = anime_diff_df, aes(x = x, y = dy, group = 1), color = "orange") + # 導関数
  geom_line(data = anime_tangentline_df, aes(x = x, y = y, group = 1), color = "turquoise4") + # 接線
  geom_point(data = anime_tangentpoint_df, aes(x = x, y = y), color = "chocolate4", size = 2) + # 接点
  geom_vline(data = anime_tangentpoint_df, aes(xintercept = x), color = "chocolate4", linetype = "dotted") + # 接線の垂線
  gganimate::transition_manual(label) + # フレーム
  ylim(c(-0.5, 1.5)) + 
  labs(title = "Sigmoid Function", 
       subtitle = "{current_frame}")

# gif画像として出力
gganimate::animate(anime_graph, nframes = length(x_vals), fps = 10)


