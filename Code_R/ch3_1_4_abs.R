
# 絶対値の劣微分 -----------------------------------------------------------------

# 利用するパッケージ
library(tidyverse)
library(gganimate)


# 絶対値の微分 ------------------------------------------------------------------

# 作図用のデータフレームを作成
df <- tidyr::tibble(
  w = seq(-1, 1, by = 0.001), # x軸の値
  abs_w = abs(w), # 絶対値
  dw = dplyr::if_else(w < 0, true = -1, false = 1) # (雑に)絶対値の微分
)

# 絶対値のグラフを作成
ggplot(df, aes(x = w, y = abs_w)) + 
  geom_line() + 
  labs(title = expression(group("|", w[k], "|")), 
       x = expression(w[k]), y = expression(group("|", w[k], "|")))

# 絶対値の微分のグラフを作成
ggplot(df, aes(x = w, y = dw)) + 
  geom_line() + 
  labs(title = expression(frac(d * group("|", w[k], "|"), d * w[k])), 
       x = expression(w[k]), y = expression(d * w[k]))



# 劣微分と傾き ------------------------------------------------------------------

# 計算用のデータフレームを作成
df <- tidyr::tibble(
  w = seq(-1, 1, by = 0.01), # x軸の値
  abs_w = abs(w) # 絶対値
)

# 絶対値の接線を計算
anime_df <- tidyr::tibble()
for(tmp_w in seq(- 1, 1, by = 0.01)) { # 接点に利用する値を指定
  if(tmp_w != 0){ # 微分可能な範囲
    # 接線を計算
    tmp_df <- df %>% 
      dplyr::mutate(
        tangent_point_w = tmp_w, # 接点のx軸の値
        tangent_point_y = abs(tmp_w), # 接点のy軸の値
        dw = dplyr::case_when(
          tmp_w > 0 ~ 1, 
          tmp_w < 0 ~ -1
        ), # 絶対値の微分(接線の傾き)
        tangent_line_y = dplyr::case_when(
          tmp_w > 0 ~ w, 
          tmp_w < 0 ~ -w
        ), # 接線のy軸の値
        label = paste0("w=", round(tmp_w, 2), ", dw=", dw) %>% 
          as.factor() # フレーム切替用のラベル
      )
    
    # 結果を結合
    anime_df <- rbind(anime_df, tmp_df)
    
  } else if(tmp_w == 0) { # 劣微分の範囲
    
    # 劣微分の値(傾き)ごとに接線を計算
    for(tmp_dw in seq(-1, 1, by = 0.01)) { # 劣微分に利用する値を指定
      # 接線を計算
      tmp_df <- df %>% 
        dplyr::mutate(
          tangent_point_w = tmp_w, # 接点のx軸の値
          tangent_point_y = abs(tmp_w), # 接点のy軸の値
          dw = tmp_dw, # 絶対値の微分(接線の傾き)
          tangent_line_y = tmp_dw * w, # 接線のy軸の値
          label = paste0("w=", round(tmp_w, 2), ", dw=", round(tmp_dw, 2)) %>% 
            as.factor() # フレーム切替用のラベル
        )
      
      # 結果を結合
      anime_df <- rbind(anime_df, tmp_df)
    }
  }
}

# 接線を作図
anime_graph <- ggplot(anime_df, aes(x = w)) + 
  geom_line(aes(y = abs_w)) + # 絶対値
  geom_point(aes(x = tangent_point_w, y = tangent_point_y), color = "red") + # 接点
  geom_line(aes(y = tangent_line_y), color = "purple", linetype = "dashed") + # 接線
  gganimate::transition_manual(label) + # フレーム
  ylim(c(-0.5, 1)) + # y軸の表示範囲
  labs(title = "Tangent Line", 
       subtitle = paste0("{current_frame}"), 
       x = expression(w[k]), y = expression(group("|", w[k], "|")))

# gif画像に変換
gganimate::animate(anime_graph, nframes = length(unique(anime_df[["label"]])), fps = 25)


