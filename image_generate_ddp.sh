
#EDT-S with AMM (cfg=1.0,2.0)
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-S/2 --amm --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz

#EDT-S without AMM (cfg=1.0,2.0)
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-S/2 --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz

#EDT-B with AMM (cfg=1.0,2.0)
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-B/2 --amm --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz

#EDT-B without AMM (cfg=1.0,2.0)
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-B/2 --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz

#EDT-XL with AMM (cfg=1.0,2.0,3.0)
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-XL/2 --amm --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz

#EDT-XL without AMM (cfg=1.0,2.0,3.0)
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-XL/2 --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet256_labeled.npz /path/save/samples/samples_50000x256x256x3.npz


#EDT-S-512 without AMM
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --model EDT-S/2 --image-size 512 --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet512_labeled.npz /path/save/samples/samples_50000x512x512x3.npz

#EDT-S-512 with AMM
torchrun --nnodes=1 --nproc_per_node=8 image_generator_ddp.py --amm --model EDT-S/2 --image-size 512 --output-dir /path/save/samples --num-images 50000 --ckpt /path/save/checkpoint.pt
python evaluator.py ./data_for_evaluation/VIRTUAL_imagenet512_labeled.npz /path/save/samples/samples_50000x512x512x3.npz