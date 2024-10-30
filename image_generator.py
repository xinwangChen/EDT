"""
Generate images from a pre-trained EDT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from load_model import find_model
from models_edt import EDT_models
import argparse
import os
# from archs.restormer_arch import Restormer


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # Load model:
    flag = ''
    if args.amm:
        model_name = args.model
        flag = '-amm'
    else:
        model_name = args.model+'_noAMM'

    latent_size = args.image_size // 8
    model = EDT_models[model_name](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # model = model = Restormer().to(device)
    # Auto-download a pre-trained model or load a custom EDT checkpoint from train.py:
    ckpt_path = args.ckpt or f"EDT-XL-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(model.get_flops())

    # print(model)
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"./sd-vae-ft-{args.vae}").to(device)

    dir_path = f"./visual/seed{args.seed}-cfg{args.cfg_scale}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Labels to condition the model with (feel free to change):
    classes = {'shark':2,'tiger_shark':3,'sea_anemone':108,'dog1':230,'dog2':232,'cat':281,'goldfinch':11,'magpie':18,'titmouse':19}

    for k,v in classes.items():

        class_labels = [v]*5

        # # Create sampling noise:
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        y = torch.tensor(class_labels, device=device)

        # # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(samples, f"./visual/seed{str(args.seed)}-cfg{str(args.cfg_scale)}/{k}{flag}.png", nrow=5, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EDT_models.keys()), default="EDT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--amm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default='./output/checkpoints.pt',
                        help="Optional path to an EDT checkpoint.")
    args = parser.parse_args()
    args.amm = True
    main(args)
    args.amm = False
    main(args)
