#small
accelerate launch --multi_gpu --num_processes 8 --main_process_port 6001 train_adan.py --image-size 256 --results-dir ./output --resume-step 0 --model EDT-S/2 --feature-path ./features256 --epochs 81
#base
accelerate launch --multi_gpu --num_processes 8 --main_process_port 6001 train_adan.py --image-size 256 --results-dir ./output --resume-step 0 --model EDT-B/2 --feature-path ./features256 --epochs 81
#extra large
accelerate launch --multi_gpu --num_processes 8 --main_process_port 6001 train_adan.py --image-size 256 --results-dir ./output --resume-step 0 --model EDT-XL/2 --init-lr 5e-4 --feature-path ./features256 --epochs 81