import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import depth_pro

def preprocess_frame(frame: np.ndarray, transform, device=None):
    frame_rgb = frame[..., ::-1] if frame.shape[2] == 3 else frame
    img_pil = Image.fromarray(frame_rgb)
    image_tensor = transform(img_pil).unsqueeze(0)
    return image_tensor.to(device) if device else image_tensor

def generate_stereo_video(input_video):
    video_name = str(Path(input_video).stem)
    output_folder_path = Path("./output/") / video_name
    output_folder_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"動画の読み込みに失敗しました: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    frame_idx = 0

    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # RGB画像として保存するための変換
            frame_rgb = frame[..., ::-1]  # BGR → RGB
            image_tensor = preprocess_frame(frame, transform, device=device)

            with torch.no_grad():
                prediction = model.infer(image_tensor, f_px=None)

            # 深度画像作成
            depth = prediction["depth"].squeeze().cpu().numpy()
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_normalized)

            # カラー画像をPIL形式で
            color_image = Image.fromarray(frame_rgb)

            # 保存パスの設定（深度＋カラー）
            depth_path = output_folder_path / f"{video_name}_depth_{frame_idx:06d}.png"
            color_path = output_folder_path / f"{video_name}_rgb_{frame_idx:06d}.png"

            # 保存
            depth_image.save(depth_path)
            color_image.save(color_path)

            frame_idx += 1
            pbar.update(1)

    cap.release()

# 実行例
if __name__ == "__main__":
    input_video = "night.mp4"
    generate_stereo_video(input_video)