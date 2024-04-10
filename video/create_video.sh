# from https://superuser.com/a/1747131
ffmpeg -f image2 -r 6 -pattern_type glob -i 'pngs/*.png' -vcodec libx264 -crf 25 video1.mp4
