#Key flags:                                                                                                                                                         
#                                                                                                                                                                    #  ┌──────────────┬─────────────────────┬───────────────────────────────────────┐                                                                              
#│     Flag     │       Default       │              Description              │                                                                                    
#
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --bindir     │ required            │ Path to bin/ directory                │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --outdir     │ <bindir>/../videos/ │ Output directory                      │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --variables  │ all                 │ Subset of variable groups             │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --fields     │ all                 │ Subset of fields                      │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --snapshots  │ all                 │ Slice like 0:5 or 10:100:2            │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --resolution │ native              │ Downsample to NxN (e.g., 1024 or 512) │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --fps        │ 30                  │ Video framerate                       │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --dpi        │ 150                 │ Frame DPI                             │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --cmap       │ auto                │ Colormap override                     │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --vscale     │ auto                │ Force linear/log/symmetric            │
#  ├──────────────┼─────────────────────┼───────────────────────────────────────┤
#  │ --workers    │ 1                   │ Parallel frame rendering              │
#  └──────────────┴─────────────────────┴───────────────────────────────────────┘
#
#  Quick test:
#  source load.sh && module load ffmpeg
#  python plotting/make_video.py \
#    --bindir /mnt/home/mgoldstein/ceph/athenak/feb13/bin \
#    --fields jz --snapshots 0:5 --resolution 512
#
#  Full run:
#  python plotting/make_video.py \
#    --bindir /mnt/home/mgoldstein/ceph/athenak/feb13/bin \
#    --resolution 1024 --workers 4

#source load.sh
#module load ffmpeg
python plotting/make_video.py --bindir /mnt/home/mgoldstein/ceph/athenak/feb13/bin  --workers 4 --field bcc1
