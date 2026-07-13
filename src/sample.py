import os
from dataclasses import dataclass
from time import time

import torch
from einops import rearrange
from PIL import ExifTags, Image
from tqdm import tqdm

from flux.ideas import denoise_cache
from flux.sampling import denoise_test_FLOPs, get_noise, get_schedule, prepare, unpack
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    prompts: list[str]
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None
    num_images_per_prompt: int
    batch_size: int
    model_name: str
    output_dir: str
    add_sampling_metadata: bool
    use_nsfw_filter: bool
    test_FLOPs: bool
    interval: int
    device: int


def main(opts: SamplingOptions):
    device = torch.device(f"cuda:{opts.device}" if torch.cuda.is_available() else "cpu")

    nsfw_classifier = None
    if opts.use_nsfw_filter:
        from transformers import pipeline

        nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection",
            device=device,
        )

    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    output_name = os.path.join(opts.output_dir, "img_{idx}.jpg")
    os.makedirs(opts.output_dir, exist_ok=True)
    idx = 0

    t5 = load_t5(device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(device)
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)

    base_seed = opts.seed if opts.seed is not None else torch.randint(0, 2**32, (1,)).item()
    total_images = len(opts.prompts) * opts.num_images_per_prompt
    progress_bar = tqdm(total=total_images, desc="Generating images")

    num_prompt_batches = (len(opts.prompts) + opts.batch_size - 1) // opts.batch_size
    sum_time = 0.0
    count = 0

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(opts.prompts))
        batch_prompts = opts.prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        for _ in range(opts.num_images_per_prompt):
            seed = base_seed + idx
            idx += num_prompts_in_batch

            x = get_noise(
                num_prompts_in_batch,
                opts.height,
                opts.width,
                device=device,
                dtype=torch.bfloat16,
                seed=seed,
            )
            inp = prepare(t5, clip, x, prompt=batch_prompts)
            timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell"))

            start_time = time()
            with torch.no_grad():
                if opts.test_FLOPs:
                    denoise_test_FLOPs(model, **inp, timesteps=timesteps, guidance=opts.guidance)
                    continue

                x = denoise_cache(
                    model,
                    **inp,
                    timesteps=timesteps,
                    guidance=opts.guidance,
                    interval=opts.interval,
                )
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

            sum_time += time() - start_time
            count += 1

            x = embed_watermark(x.clamp(-1, 1).float())
            x = rearrange(x, "b c h w -> b h w c")

            for i in range(num_prompts_in_batch):
                img = Image.fromarray((127.5 * (x[i] + 1.0)).cpu().byte().numpy())

                if opts.use_nsfw_filter:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next((res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0)
                else:
                    nsfw_score = 0.0

                if nsfw_score >= NSFW_THRESHOLD:
                    print("Generated image may contain inappropriate content, skipped.")
                    progress_bar.update(1)
                    continue

                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                exif_data[ExifTags.Base.Model] = model_name
                if opts.add_sampling_metadata:
                    exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]

                fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                img.save(fn, exif=exif_data, quality=95, subsampling=0)
                progress_bar.update(1)

    if count > 0:
        print(f"Average generation time: {sum_time / count:.2f}s")
    progress_bar.close()


def read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images with L2P-accelerated FLUX.")
    parser.add_argument("--prompt_file", type=str, default="predictor/DrawBench200.txt")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="flux-dev", choices=["flux-dev", "flux-schnell"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--add_sampling_metadata", action="store_true")
    parser.add_argument("--use_nsfw_filter", action="store_true")
    parser.add_argument("--test_FLOPs", action="store_true")
    parser.add_argument("--interval", type=int, default=7, help="L2P cache interval.")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    main(
        SamplingOptions(
            prompts=read_prompts(args.prompt_file),
            width=args.width,
            height=args.height,
            num_steps=args.num_steps,
            guidance=args.guidance,
            seed=args.seed,
            num_images_per_prompt=args.num_images_per_prompt,
            batch_size=args.batch_size,
            model_name=args.model_name,
            output_dir=args.output_dir,
            add_sampling_metadata=args.add_sampling_metadata,
            use_nsfw_filter=args.use_nsfw_filter,
            test_FLOPs=args.test_FLOPs,
            interval=args.interval,
            device=args.device,
        )
    )


if __name__ == "__main__":
    app()
