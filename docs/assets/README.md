# 本项目实机演示

`pi05_step45000_4x.mp4` 与 `pi05_step45000_4x.gif` 来自本项目参与者提供的手机实拍 `docs/IMG_0596.MOV`，展示45000步检查点的真机夹取过程。原片约127.6秒，展示版约31.9秒，全片4倍速、无裁切、无音频。README使用循环GIF，并链接H.264 MP4。导出时移除源文件元数据，原始MOV不提交。

从仓库根目录可使用FFmpeg重建（保留原片旋转方向，输出竖屏）：

```bash
ffmpeg -i docs/IMG_0596.MOV -map 0:v:0 -an \
  -vf 'setpts=0.25*(PTS-STARTPTS),scale=540:-2' -r 24 -vsync vfr \
  -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p \
  -map_metadata -1 -metadata:s:v:0 rotate=0 -movflags +faststart \
  docs/assets/pi05_step45000_4x.mp4

ffmpeg -i docs/assets/pi05_step45000_4x.mp4 \
  -filter_complex 'fps=6,scale=280:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none' \
  -loop 0 -map_metadata -1 docs/assets/pi05_step45000_4x.gif
```

单段演示不等于统计成功率。这里的视频属于本次实验，与上游ACT演示不同。
