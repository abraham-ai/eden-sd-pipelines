
import sys, os, shutil
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from eden_utils import *
from depth_transforms import *
from PIL import Image, ImageOps

def extract_frames_from_video(video_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the fps using ffprobe
    ffprobe_cmd = f"ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {video_path}"
    fps_output = subprocess.check_output(ffprobe_cmd, shell=True).decode().strip()
    fps = eval(fps_output)  # Convert the string "num/den" to a float

    # Extract frames using ffmpeg
    print(f"Extracting frames from {video_path} to {output_dir} at {fps} fps...")
    ffmpeg_cmd = f"ffmpeg -i {video_path} -vf fps={fps} {output_dir}/frame_%06d.jpg"
    os.system(ffmpeg_cmd)

    return fps

def predict_depth_map_zoe(pil_image, zoe, depth_rescale, flip_aug = False):
    min_v, max_v = depth_rescale

    depth_tensor = zoe.infer_pil(pil_image, output_type="tensor")
    if flip_aug:
        flipped_tensor = zoe.infer_pil(ImageOps.mirror(pil_image), output_type="tensor")
        depth_tensor = 0.5 * (depth_tensor + torch.flip(flipped_tensor, dims=[1]))

    # renormalize depth map:
    depth_tensor  = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min()) * (max_v - min_v) + min_v
    pil_depth_map = Image.fromarray(depth_tensor.permute(0, 1).cpu().numpy().astype(np.uint8))

    return np.array(pil_depth_map), depth_tensor

def predict_depth_map_midas(pil_image, depth_estimator, feature_extractor, depth_rescale):
    min_v, max_v = depth_rescale

    width, height = pil_image.size
    image = feature_extractor(images=pil_image, return_tensors="pt").pixel_values.to("cuda")

    depth_tensor = depth_estimator(image).predicted_depth

    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    depth_tensor = 1 - (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
    depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min()) * (max_v - min_v) + min_v

    pil_depth_map = Image.fromarray((depth_tensor.permute(0, 2, 3, 1).cpu().numpy()[0].squeeze()).astype(np.uint8))

    return np.array(pil_depth_map), depth_tensor

def blend_pil_imgs(pil_img1, pil_img2, alpha):
    img1 = np.array(pil_img1)
    img2 = np.array(pil_img2)
    return Image.fromarray((alpha * img1 + (1 - alpha) * img2).astype(np.uint8))

def extract_depth(input_path, 
        render_video = False, 
        depth_type = "midas",
        flip_aug = False,
        temporal_smoothing = False,
        depth_rescale = [0., 255.]
        ):
    fps = 24
    if input_path.endswith(".mp4"):
        frames_dir = os.path.join(os.path.dirname(input_path), "frames")
        fps = extract_frames_from_video(input_path, frames_dir)
    elif os.path.isdir(input_path):
        print(f"Found {len(os.listdir(input_path))} files in {input_path}..")
        frames_dir = input_path
    else:
        raise ValueError("Invalid input path! (needs to be a directory or a video file)")

    output_dir = os.path.join(frames_dir, "depth_maps")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    torch.cuda.empty_cache()
    if depth_type == "midas":
        depth_model       = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
        feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    else:
        repo = "isl-org/ZoeDepth"
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
        depth_model = torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(settings._device)

    frame_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg") and "_depth" not in f])
    print("Found %d frames!" % len(frame_paths))

    prev_depth_maps = []
    avg_weights = [1.0, 0.75, 0.5, 0.25]

    with torch.no_grad():
        print("Predicting depth maps...")
        for i, frame_path in tqdm(enumerate(frame_paths)):

            frame = Image.open(frame_path)

            if depth_type == "midas":
                depth_map, depth_tensor = predict_depth_map_midas(frame, depth_model, feature_extractor, [0., 255.])
                
                if flip_aug:
                    flipped_depth_map, _ = predict_depth_map_midas(ImageOps.mirror(frame), depth_model, feature_extractor, [0., 255.])
                    depth_map = 0.5 * (depth_map + flipped_depth_map[:, ::-1])
            else:
                depth_map, depth_tensor = predict_depth_map_zoe(frame, depth_model)
                if flip_aug:
                    flipped_depth_map, _ = predict_depth_map_zoe(ImageOps.mirror(frame), depth_model)
                    depth_map = 0.5 * (depth_map + flipped_depth_map[:, ::-1])

            prev_depth_maps.append(depth_map)

            if temporal_smoothing:
                n_frames_to_avg = min(len(prev_depth_maps), len(avg_weights))
                normalized_weights = np.array(avg_weights[:n_frames_to_avg]) / sum(avg_weights[:n_frames_to_avg])
                depth_map = np.average(prev_depth_maps[-n_frames_to_avg:], axis=0, weights=normalized_weights)

            prev_depth_maps.append(depth_map)
            depth_map = Image.fromarray(depth_map.astype(np.uint8))
            depth_map.save(os.path.join(output_dir, os.path.basename(frame_path)), quality=95)

    if render_video:
        print("Rendering depth maps to video...")
        output_video_path = os.path.join(os.path.dirname(frames_dir), "depth_video.mp4")
        ffmpeg_cmd = f"ffmpeg -r {fps} -f image2 -pattern_type glob -i '{output_dir}/*.jpg' -vcodec libx264 -pix_fmt yuv420p {output_video_path}"
        os.system(ffmpeg_cmd)
        return output_video_path

    else:
        return output_dir


if __name__ == "__main__":
    input_path = "/data/xander/Projects/cog/GitHub_repos/eden-comfyui/tests/videos/road.mp4"
    output_path = extract_depth(input_path, 
        render_video=True, depth_type="midas",
        temporal_smoothing=True, flip_aug=False)

    print(f"Depth maps saved to {output_path}")