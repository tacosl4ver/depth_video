import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import subprocess
import glob

def extract_fps_and_audio(source_video, temp_audio_path="temp_audio.aac"):
    # FPS取得
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", source_video],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    fps_text = probe.stdout.strip()
    if '/' in fps_text:
        num, den = map(int, fps_text.split('/'))
        fps = num / den
    else:
        fps = float(fps_text)

    # 音声抽出
    subprocess.run([
        "ffmpeg", "-y", "-i", source_video, "-vn", "-acodec", "copy", temp_audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return fps, temp_audio_path

def apply_horizontal_shift(rgb_img, shift_map):
    """
    深度シフトをNumPyのベクトル演算で高速に適用
    """
    h, w, c = rgb_img.shape
    x_coords = np.arange(w)[None, :]  # shape: (1, w)
    x_map = np.clip(x_coords - shift_map, 0, w - 1)
    indices_y = np.arange(h)[:, None]
    shifted_rgb = rgb_img[indices_y, x_map]
    return shifted_rgb

def save_video_directly(input_folder, output_path, fps, max_pixel_shift):
    input_folder = Path(input_folder)
    rgb_files = sorted(glob.glob(str(input_folder / "*_rgb_*.png")))
    depth_files = sorted(glob.glob(str(input_folder / "*_depth_*.png")))

    assert len(rgb_files) == len(depth_files), "RGB画像とDepth画像の数が一致しません。"

    # 最初の画像でサイズ取得
    sample = np.array(Image.open(rgb_files[0]))
    h, w, _ = sample.shape
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))

    for rgb_path, depth_path in tqdm(zip(rgb_files, depth_files), total=len(rgb_files), desc="Saving stereo video"):
        rgb_img = np.array(Image.open(rgb_path))
        depth_img = np.array(Image.open(depth_path)).astype(np.float32) / 255.0

        # 近いものが黒 → 白が遠い → 白の方が大きいピクセルシフト
        shift_map = (depth_img * max_pixel_shift).astype(np.int32)
        right_eye = apply_horizontal_shift(rgb_img, shift_map)
        left_eye = rgb_img
        stereo_img = np.concatenate((left_eye, right_eye), axis=1)

        out.write(cv2.cvtColor(stereo_img, cv2.COLOR_RGB2BGR))

    out.release()

def mux_audio_with_video(video_path, audio_path, final_output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", final_output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def create_spatial_video_with_audio(input_folder, source_video, max_pixel_shift):
    print("抽出中：FPSと音声...")
    fps, audio_path = extract_fps_and_audio(source_video)
    print(f"FPS: {fps}, 音声ファイル: {audio_path}")

    input_folder = Path(input_folder)
    base_name = input_folder.name
    output_video_path = f"{base_name}_spatial_shift{max_pixel_shift}.mp4"
    temp_video_path = "temp_video.mp4"

    print("空間ビデオを生成して保存中（メモリ効率化）...")
    save_video_directly(input_folder, temp_video_path, fps, max_pixel_shift)

    print("音声を合成中...")
    mux_audio_with_video(temp_video_path, audio_path, output_video_path)

    print(f"🎬 完成：{output_video_path}")

    # 一時ファイルを削除
    Path(temp_video_path).unlink(missing_ok=True)
    Path(audio_path).unlink(missing_ok=True)

# 実行例
if __name__ == "__main__":
    create_spatial_video_with_audio(
        input_folder="./output/night",    # 前処理した画像フォルダ
        source_video="night.mp4",         # 元の動画（音声・fps抽出に使用）
        max_pixel_shift=30                # ピクセルシフト量（視差）
    )
