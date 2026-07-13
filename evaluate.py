import argparse
import os
import re

import cv2
import lpips
import numpy as np
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_prompts(prompt_file_path):
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def get_sorted_image_files(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            image_files.append(filename)

    def natural_sort_key(filename):
        match = re.search(r"img_(\d+)\.", filename)
        return int(match.group(1)) if match else 0

    return sorted(image_files, key=natural_sort_key)


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2, data_range=255)
    return ssim(img1, img2, data_range=255)


def preprocess_for_lpips(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img * 2 - 1


def evaluate_all_metrics(
    test_folder,
    prompt_file_path=None,
    reference_folder=None,
    clip_model_path=None,
    imagereward_model_path=None,
    pickscore_processor_path=None,
    pickscore_model_path=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model = None
    clip_processor = None
    imagereward_model = None
    pickscore_processor = None
    pickscore_model = None
    reward_transform = None

    if clip_model_path:
        from transformers import CLIPModel, CLIPProcessor

        clip_model = CLIPModel.from_pretrained(clip_model_path).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

    if imagereward_model_path:
        import ImageReward as RM

        med_config = os.path.join(imagereward_model_path, "med_config.json")
        imagereward_path = os.path.join(imagereward_model_path, "ImageReward.pt")
        imagereward_model = RM.load(
            imagereward_path,
            download_root=imagereward_model_path,
            med_config=med_config,
        ).to(device)
        reward_transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    if pickscore_processor_path and pickscore_model_path:
        from transformers import AutoModel, AutoProcessor

        pickscore_processor = AutoProcessor.from_pretrained(pickscore_processor_path)
        pickscore_model = AutoModel.from_pretrained(pickscore_model_path).eval().to(device)

    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device) if reference_folder else None

    image_files = get_sorted_image_files(test_folder)
    prompts = load_prompts(prompt_file_path) if prompt_file_path else []

    clip_scores = []
    imagereward_scores = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    pickscore_scores = []

    for i, filename in tqdm(enumerate(image_files), total=len(image_files), desc="Evaluating Metrics"):
        try:
            img_path = os.path.join(test_folder, filename)
            img_pil = Image.open(img_path).convert("RGB")

            if i < len(prompts) and clip_model is not None and clip_processor is not None:
                prompt = prompts[i]
                with torch.no_grad():
                    inputs = clip_processor(
                        text=prompt,
                        images=img_pil,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    outputs = clip_model(**inputs)
                    clip_scores.append(outputs.logits_per_image.item())

                if imagereward_model is not None and reward_transform is not None:
                    with torch.no_grad():
                        img_tensor = TF.pil_to_tensor(img_pil).unsqueeze(0).to(device)
                        img_reward = reward_transform(img_tensor)
                        inputs = imagereward_model.blip.tokenizer(
                            [prompt],
                            padding="max_length",
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(device)
                        score = imagereward_model.score_gard(
                            inputs.input_ids,
                            inputs.attention_mask,
                            img_reward,
                        )
                        imagereward_scores.append(score.item())

                if pickscore_processor is not None and pickscore_model is not None:
                    with torch.no_grad():
                        inputs_pickscore = pickscore_processor(
                            text=prompt,
                            images=img_pil,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        ).to(device)
                        outputs_pickscore = pickscore_model(**inputs_pickscore)
                        pickscore_scores.append(outputs_pickscore.logits_per_image.item())

            if reference_folder and lpips_model is not None:
                ref_path = os.path.join(reference_folder, filename)
                if os.path.exists(ref_path):
                    img_cv = cv2.imread(img_path)
                    ref_cv = cv2.imread(ref_path)
                    if img_cv is not None and ref_cv is not None:
                        if img_cv.shape != ref_cv.shape:
                            ref_cv = cv2.resize(ref_cv, (img_cv.shape[1], img_cv.shape[0]))

                        psnr_values.append(calculate_psnr(img_cv, ref_cv))
                        ssim_values.append(calculate_ssim(img_cv, ref_cv))

                        with torch.no_grad():
                            img_lpips = preprocess_for_lpips(img_cv).to(device)
                            ref_lpips = preprocess_for_lpips(ref_cv).to(device)
                            lpips_values.append(lpips_model(img_lpips, ref_lpips).item())
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            continue

    results = {}
    if clip_scores:
        results["clip_score"] = np.mean(clip_scores)
    if imagereward_scores:
        results["imagereward"] = np.mean(imagereward_scores)
    if pickscore_scores:
        results["pickscore"] = np.mean(pickscore_scores)
    if psnr_values:
        results["psnr"] = np.mean(psnr_values)
    if ssim_values:
        results["ssim"] = np.mean(ssim_values)
    if lpips_values:
        results["lpips"] = np.mean(lpips_values)
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified metrics evaluation")
    parser.add_argument("--test_folder", type=str, required=True, help="Test images folder")
    parser.add_argument("--prompt_file", type=str, default="predictor/DrawBench200.txt")
    parser.add_argument("--reference_folder", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--imagereward_model_path", type=str, default=None)
    parser.add_argument("--pickscore_processor_path", type=str, default=None)
    parser.add_argument("--pickscore_model_path", type=str, default=None)
    args = parser.parse_args()

    results = evaluate_all_metrics(
        test_folder=args.test_folder,
        prompt_file_path=args.prompt_file,
        reference_folder=args.reference_folder,
        clip_model_path=args.clip_model_path,
        imagereward_model_path=args.imagereward_model_path,
        pickscore_processor_path=args.pickscore_processor_path,
        pickscore_model_path=args.pickscore_model_path,
    )

    print("\nResult:(ClipScore, ImageReward, PickScore, PSNR, SSIM, LPIPS)")
    print(f"{results.get('clip_score', 0):.4f}")
    print(f"{results.get('imagereward', 0):.4f}")
    print(f"{results.get('pickscore', 0):.4f}")
    print(f"{results.get('psnr', 0):.3f}")
    print(f"{results.get('ssim', 0):.4f}")
    print(f"{results.get('lpips', 0):.4f}")


if __name__ == "__main__":
    main()
