
# ch3.1.4：Lpノルム ----------------------------------------------------------

# 3.1.4項で利用するパッケージ
library(tidyverse)
library(gganimate)


### Lpノルムのグラフ -----

# 作図用のwの範囲を指定
w_vec <- seq(-10, 10, by = 0.1)


# 値を指定
q <- 2

# Lpノルムを計算
norm_df <- tidyr::tibble(
  w_1 = rep(w_vec, times = length(w_vec)), # w1の値
  w_2 = rep(w_vec, each = length(w_vec)), # w2の値
  Lq_norm = (abs(w_1)^q + abs(w_2)^q)^(1 / q) # ノルム
)

# ノルムの等高線図を作成
ggplot(norm_df, aes(x = w_1, y = w_2, z = Lq_norm, color = ..level..)) + 
  geom_contour() + # 等高線グラフ
  #geom_contour(breaks = 1.0) + # 等高線グラフ:(値を指定)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(group("|", group("|", w, "|"), "|")[q] == sqrt(sum(group("|", w[j], "|")^q, j==1, M), q)), 
       subtitle = paste0("q=", q), 
       x = expression(w[1]), y = expression(w[2]), color = paste0("L", q, " norm"))


# 正則化項を計算
E_df <- tidyr::tibble(
  w_1 = rep(w_vec, times = length(w_vec)), # w1の値
  w_2 = rep(w_vec, each = length(w_vec)), # w2の値
  E_w = (abs(w_1)^q + abs(w_2)^q) / q # 正則化項
)

# 正則化項の等高線図を作成
ggplot(E_df, aes(x = w_1, y = w_2, z = E_w, color = ..level..)) + 
  #geom_contour() + # 等高線グラフ
  geom_contour(breaks = 1.0) + # 等高線グラフ:(値を指定)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E[w](w) == sum(group("|", w[j], "|")^q, j==1, M)), 
       subtitle = paste0("q=", q), 
       x = expression(w[1]), y = expression(w[2]), color = expression(E[w](w)))


### qと形状の関係 -----

# qの最大値を指定
q_max <- 7.5

# 使用するqの値を作成
q_vec <- seq(0.5, q_max, by = 0.1)
length(q_vec)

# qごとにノルムを計算
anime_df <- tidyr::tibble()
for(q in q_vec) {
  # Lpノルムを計算
  tmp_norm_df <- tidyr::tibble(
    q = q, 
    w_1 = rep(w_vec, times = length(w_vec)), # w1の値
    w_2 = rep(w_vec, each = length(w_vec)), # w2の値
    Lq_norm = (abs(w_1)^q + abs(w_2)^q)^(1 / q) # ノルム
  )
  
  # 結果を結合
  anime_df <- rbind(anime_df, tmp_norm_df)
}


# ノルムの等高線図を作成
anime_graph <- ggplot(anime_df, aes(x = w_1, y = w_2, z = Lq_norm, color = ..level..)) + 
  geom_contour() + # 等高線グラフ
  gganimate::transition_manual(q) + # フレーム
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(group("|", group("|", w, "|"), "|")[q] == sqrt(sum(group("|", w[j], "|")^q, j==1, M), q)), 
       subtitle = paste0("q={current_frame}"), 
       x = expression(w[1]), y = expression(w[2]), color = "Lp norm")

# gif画像に変換
gganimate::animate(anime_graph, nframes = length(q_vec), fps = 10)


