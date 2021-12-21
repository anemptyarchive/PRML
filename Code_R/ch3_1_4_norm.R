
# ch3.1.4：Lpノルム ----------------------------------------------------------

# 3.1.4項で利用するパッケージ
library(tidyverse)
library(gganimate)


### Lpノルムのグラフ -----

# 値を指定
p <- 2

# 作図用のwの範囲を指定
w_vals <- seq(-10, 10, by = 0.1)

# Lpノルムを計算
norm_df <- tidyr::tibble(
  w_1 = rep(w_vals, times = length(w_vals)), # w1の値
  w_2 = rep(w_vals, each = length(w_vals)), # w2の値
  Lp_norm = (abs(w_1)^p + abs(w_2)^p)^(1 / p) # ノルム
)

# ノルムの等高線図を作成
ggplot(norm_df, aes(x = w_1, y = w_2, z = Lp_norm, color = ..level..)) + 
  geom_contour() + # 等高線グラフ
  #geom_contour(breaks = 1.0) + # 等高線グラフ:(値を指定)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(group("|", group("|", w, "|"), "|")[p] == sqrt(sum(group("|", w[j], "|")^p, j==1, M), p)), 
       subtitle = paste0("p=", p), 
       x = expression(w[1]), y = expression(w[2]), color = paste0("L", p, " norm"))


# 正則化項を計算
E_df <- tidyr::tibble(
  w_1 = rep(w_vals, times = length(w_vals)), # w1の値
  w_2 = rep(w_vals, each = length(w_vals)), # w2の値
  E_W = (abs(w_1)^p + abs(w_2)^p) / p # 正則化項
)

# 正則化項の等高線図を作成
ggplot(E_df, aes(x = w_1, y = w_2, z = E_W, color = ..level..)) + 
  #geom_contour() + # 等高線グラフ
  geom_contour(breaks = 1.0) + # 等高線グラフ:(値を指定)
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = expression(E[w](w) == sum(group("|", w[j], "|")^p, j==1, M)), 
       subtitle = paste0("p=", p), 
       x = expression(w[1]), y = expression(w[2]), color = expression(E[w](w)))


### pと形状の関係 -----

# 使用するpの値を作成
p_vals <- seq(0.5, 7.5, by = 0.1)
length(p_vals) # フレーム数

# pごとにノルムを計算
anime_norm_df <- tidyr::tibble()
for(p in p_vals) {
  # Lpノルムを計算
  tmp_norm_df <- tidyr::tibble(
    p = p, 
    w_1 = rep(w_vals, times = length(w_vals)), # w1の値
    w_2 = rep(w_vals, each = length(w_vals)), # w2の値
    Lp_norm = (abs(w_1)^p + abs(w_2)^p)^(1 / p) # ノルム
  )
  
  # 結果を結合
  anime_norm_df <- rbind(anime_norm_df, tmp_norm_df)
}


# ノルムの等高線図を作成
anime_graph <- ggplot(anime_norm_df, aes(x = w_1, y = w_2, z = Lp_norm, color = ..level..)) + 
  geom_contour() + # 等高線グラフ
  gganimate::transition_manual(p) + # フレーム
  coord_fixed(ratio = 1) + # アスペクト比
  labs(title = "Lp-Norm", 
       subtitle = paste0("p={current_frame}"), 
       x = expression(w[1]), y = expression(w[2]), color = "Lp norm")

# gif画像に変換
gganimate::animate(anime_graph, nframes = length(p_vals), fps = 10)


