#
# SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0
#

import os
import sys
import csv
import shlex
import shutil
import argparse
import importlib
import datetime as dt
from tqdm import tqdm
import subprocess as sp
from omegaconf import OmegaConf
from termcolor import colored

import cv2
import numpy as np
from PIL import Image
import matplotlib.cm as cm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.modeling.sharingan import Sharingan
from src.utils.common import spatial_argmax2d, square_bbox

from boxmot import DeepOcSort as DeepOCSORT, ByteTrack as BYTETracker, OcSort as OCSORT




# ================================ ARGS ================================ #
parser = argparse.ArgumentParser(description="Predict gaze on videos")
parser.add_argument("--input-dir", type=str, default="data", help="Name of the folder where to find the input.")
parser.add_argument("--input-filename", type=str, help="Name of the clip file to process (with extension).")
parser.add_argument("--output-dir", type=str, default="data", help="Name of the folder where to save the output.")
parser.add_argument("--heatmap-pid", type=int, default=-1, help="pid of the person to draw the heatmap of.")

parser.add_argument("--filter-by-inout", action='store_true', help="Whether to hide the gaze point when inout < 0.5.")
parser.add_argument('--no-filter-by-inout', dest='filter_by_inout', action='store_false')
parser.set_defaults(filter_by_inout=False)

parser.add_argument("--show-gaze-vec", action='store_true', help="Whether to draw the gaze vector.")
parser.add_argument('--no-show-gaze-vec', dest='show_gaze_vec', action='store_false')
parser.set_defaults(show_gaze_vec=False)

parser.add_argument("--write-video", action='store_true', default=False, help="Also write an annotated output video (default: CSV only).")
parser.add_argument("--io-thr", type=float, default=0.5, help="In/out score threshold for gaze target assignment.")

args = parser.parse_args()


# =============================== GLOBALS =============================== #
TERM_COLOR = "cyan"
COLOR_NAMES = ["mediumvioletred", "green", "dodgerblue", "crimson", "goldenrod", "DarkSlateGray",
			   "saddlebrown", "purple", "teal"]
COLORS = [(199, 21, 133), (0, 128, 0), (30, 144, 255), (220, 20, 60), (218, 165, 32),
		  (47, 79, 79), (139, 69, 19), (128, 0, 128), (0, 128, 128)]

DET_THR = 0.4 # head detection threshold
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]

CKPT_PATH = "checkpoints/videoattentiontarget.pt"


def get_device():
	if not torch.cuda.is_available():
		return torch.device("cpu")

	try:
		device = torch.device("cuda")
		# Probe with a real CUDA op instead of checking arch tokens only.
		_ = torch.zeros(1, device=device)
		torch.cuda.synchronize()
		return torch.device("cuda")
	except Exception as exc:
		name = "unknown"
		arch = "unknown"
		try:
			props = torch.cuda.get_device_properties(0)
			name = props.name
			arch = f"sm_{props.major}{props.minor}"
		except Exception:
			pass

		print(colored(
			f"CUDA device {name} (compute {arch}) failed a runtime probe with this PyTorch build ({exc}). Falling back to CPU.",
			"red",
		))
		return torch.device("cpu")

DEVICE = get_device()
print(colored(f"Using device: {DEVICE}", TERM_COLOR))

# ========================= UTILITY FUNCTIONS =========================== #
def resolve_gaze_targets(pids, head_bboxes, gaze_points, inouts, img_w, img_h, io_thr=0.5):
	"""
	For each detected person, determine which other participant (if any) they
	are looking at by checking whether the gaze point falls inside another
	person's head bbox. Returns a list of dicts (one per detected person).
	"""
	rows = []
	if len(pids) == 0:
		return rows

	# Convert tensors to numpy
	hb = head_bboxes.numpy() if isinstance(head_bboxes, torch.Tensor) else np.array(head_bboxes)
	gp = gaze_points.numpy() if isinstance(gaze_points, torch.Tensor) else np.array(gaze_points)
	io = inouts.numpy() if isinstance(inouts, torch.Tensor) else np.array(inouts)

	# Gaze points are normalised 0..1 → pixel coords
	if gp.size > 0 and gp.max() <= 1.0:
		gp = gp * np.array([img_w, img_h])

	for i, pid in enumerate(pids):
		row = {
			"looker_pid": int(pid),
			"target_pid": "",
		}

		# Skip target assignment when gaze data missing or inout below threshold
		if gp.size == 0 or (float(io[i]) if io.size > 0 else 0.0) < io_thr:
			rows.append(row)
			continue

		gx, gy = gp[i]
		candidates = []
		for j, other_pid in enumerate(pids):
			if other_pid == pid:
				continue
			xmin, ymin, xmax, ymax = hb[j]
			if xmin <= gx <= xmax and ymin <= gy <= ymax:
				cx = (xmin + xmax) / 2
				cy = (ymin + ymax) / 2
				dist = (gx - cx) ** 2 + (gy - cy) ** 2
				candidates.append((int(other_pid), dist))

		if candidates:
			candidates.sort(key=lambda c: c[1])
			row["target_pid"] = candidates[0][0]

		rows.append(row)
	return rows


def expand_bbox(bbox, img_w, img_h, k=0.1):
	w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
	bbox[0] = max(0, bbox[0] - k * w)
	bbox[1] = max(0, bbox[1] - k * h)
	bbox[2] = min(img_w, bbox[2] + k * w)
	bbox[3] = min(img_h, bbox[3] + k * h)
	return bbox

def load_tracker():
	#tracker = DeepOCSORT(
	#  model_weights=Path('/idiap/temp/stafasca/weights/tracking/osnet_x0_25_msmt17.pt'),  # which ReID model to use
	#  device=self.device,  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
	#  fp16=True,  # wether to run the ReID model with half precision or not
	#)
	#tracker = BYTETracker()
	tracker = OCSORT()
	return tracker


def patch_yolov5_autocast(model):
	"""Switch YOLOv5 AutoShape to torch.amp.autocast(device_type='cuda', ...)."""
	if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
		return

	try:
		module = importlib.import_module(model.__class__.__module__)
		amp_module = getattr(module, "amp", None)
		if amp_module is None:
			return

		def autocast_cuda(enabled=True, *args, **kwargs):
			kwargs = dict(kwargs)
			kwargs.setdefault("enabled", enabled)
			return torch.amp.autocast("cuda", *args, **kwargs)

		autocast_cuda.__name__ = "autocast_cuda"
		amp_module.autocast = autocast_cuda
	except Exception:
		# Keep demo robust even if the internal YOLOv5 module layout changes.
		return

def load_head_detection_model(device):
	# Load and return the pre-trained head detection model
	ckpt_path = "./weights/yolov5m_crowdhuman.pt"
	if isinstance(device, torch.device):
		device_str = "cpu" if device.type == "cpu" else str(device.index or 0)
	else:
		device_str = str(device)
	model = torch.hub.load("ultralytics/yolov5", "custom", path=ckpt_path, verbose=False, device=device_str)
	patch_yolov5_autocast(model)
	model.conf = 0.25  # NMS confidence threshold
	model.iou = 0.45  # NMS IoU threshold
	model.classes = [1]  # filter by class, i.e. = [1] for heads
	model.amp = False  # Automatic Mixed Precision (AMP) inference
	model = model.to(device)
	model.eval()
	return model

def detect_heads(image, model):
	"""
	Detect heads in the image using the provided model.
	Returns a numpy array containing the detected head bboxes and their confidence scores.
	"""
	# Convert BGR OpenCV image to RGB and keep HWC numpy format for YOLO AutoShape
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = np.ascontiguousarray(image)

	results = model(image, size=640)
	detections = results.xyxy[0]
	if detections is None or detections.shape[0] == 0:
		return np.zeros((0, 5), dtype=np.float32)

	return detections[:, :5].cpu().numpy()


def get_ffmpeg_path():
	# Prefer a working ffmpeg binary outside the current conda env if needed.
	candidates = [
		shutil.which("ffmpeg"),
		"/home/linuxbrew/.linuxbrew/bin/ffmpeg",
		"/usr/bin/ffmpeg",
	]
	for candidate in candidates:
		if not candidate:
			continue
		if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
			try:
				sp.run([candidate, "-version"], stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)
				return candidate
			except Exception:
				continue
	raise RuntimeError("No working ffmpeg binary found. Install ffmpeg or set FFMPEG_PATH.")


def load_sharingan_model(ckpt_path, device):
	# Build model
	sharingan = Sharingan(
		patch_size=16,
		token_dim=768,
		image_size=224,
		gaze_feature_dim=512,
		encoder_depth=12,
		encoder_num_heads=12,
		encoder_num_global_tokens=0,
		encoder_mlp_ratio=4.0,
		encoder_use_qkv_bias=True,
		encoder_drop_rate=0.0,
		encoder_attn_drop_rate=0.0,
		encoder_drop_path_rate=0.0,
		decoder_feature_dim=128,
		decoder_hooks=[2, 5, 8, 11],
		decoder_hidden_dims=[48, 96, 192, 384],
		decoder_use_bn=True,
	)

	# Load checkpoint
	checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
	checkpoint = {name.replace("model.", ""): value for name, value in checkpoint["state_dict"].items()}
	sharingan.load_state_dict(checkpoint, strict=True)
	sharingan.eval()
	sharingan.to(device)
	return sharingan

def predict_gaze(image, sharingan, head_detector, tracker=None):
	# 1. Convert image
	image_np = np.array(image)
	img_h, img_w, img_c = image_np.shape

	raw_detections = detect_heads(image_np, head_detector)
	detections = []
	for k, raw_detection in enumerate(raw_detections):
		bbox, conf = raw_detection[:4], raw_detection[4]
		if conf > DET_THR:
			#bbox = expand_bbox(bbox, img_w, img_h, k=0.1)
			cls_ = np.array([0.])
			detection = np.concatenate([bbox, conf[None], cls_])
			detections.append(detection)
	if len(detections) == 0:
		return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
	detections = np.stack(detections)
	num_heads = len(detections)

	# 2. Detect & track head bboxes
	tracks = tracker.update(detections, image_np)
	if len(tracks) == 0: # sometimes tracker.update returns [] even when detections is not []
		return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
	pids = (tracks[:, 4] - 1).astype(int)
	head_bboxes = torch.from_numpy(tracks[:, :4]).float()
	t_head_bboxes = square_bbox(head_bboxes, img_w, img_h)

	# 3. Extract and transform heads
	heads = []
	for bbox in t_head_bboxes:
		head = TF.resize(TF.to_tensor(image.crop(bbox.numpy())), (224, 224))
		heads.append(head)
	heads = torch.stack(heads)
	heads = TF.normalize(heads, mean=IMG_MEAN, std=IMG_STD)

	# 4. Transform Image
	image = TF.to_tensor(image)
	image = TF.resize(image, (224, 224))
	image = TF.normalize(image, mean=IMG_MEAN, std=IMG_STD)

	# 5. Normalize head bboxes
	scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
	t_head_bboxes /= scale

	# 6. build input sample
	sample = {}
	sample["image"] = image.unsqueeze(0).to(DEVICE) # (1, 3, 224, 224)
	sample["heads"] = heads.unsqueeze(0).to(DEVICE) # (1, num_heads, 3, 224, 224)
	sample["head_bboxes"] = t_head_bboxes.unsqueeze(0).to(DEVICE) # (1, num_heads, 4)

	# 7. predict gaze
	with torch.no_grad():
		gaze_vecs, gaze_heatmaps, inouts = sharingan(sample)
		gaze_heatmaps = gaze_heatmaps.squeeze(0).cpu()
		gaze_vecs = gaze_vecs.squeeze(0).cpu()
		gaze_points = spatial_argmax2d(gaze_heatmaps, normalize=True)
		inouts = torch.sigmoid(inouts.squeeze(0)).flatten().cpu()

	return gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids

def draw_gaze(
	image,
	head_bboxes,
	gaze_points,
	gaze_vecs,
	inouts,
	pids,
	gaze_heatmaps,
	heatmap_pid = None,
	frame_nb = None,
	colors = COLORS,
	filter_by_inout = False,
	alpha: float = 0.5,
	io_thr: float = 0.5,
	gaze_pt_size: int = 10,
	gaze_vec_factor: float = 0.8,
	head_center_size: int = 10,
	thickness: int = 4,
	fs: float = 0.6,
):
	"""
	Draws gaze results on the given image.

	Args:
		image (np.ndarray): The input image on which to draw.
		head_bboxes (array-like): Bounding boxes for heads.
		gaze_points (array-like): Points representing gaze locations.
		gaze_vecs (array-like): Vectors representing gaze directions.
		inouts (array-like): In/out scores for each head.
		pids (array-like): Person IDs for each head.
		gaze_heatmaps (array-like): Heatmaps for gaze.
		heatmap_pid (int, optional): Person ID for which to draw the heatmap. Defaults to None.
		frame_nb (int, optional): Frame number to display on the image. Defaults to None.
		colors (array-like, optional): Colors to use for drawing. Defaults to COLORS.
		alpha (float, optional): Alpha blending value for heatmap overlay. Defaults to 0.5.
		io_thr (float, optional): Threshold for in/out scores to draw gaze points. Defaults to 0.5.
		gaze_pt_size (int, optional): Size of the gaze points. Defaults to 10.
		gaze_vec_factor (float, optional): Scaling factor for gaze vectors. Defaults to 0.8.
		head_center_size (int, optional): Size of the head center points. Defaults to 10.
		thickness (int, optional): Thickness of the drawing lines. Defaults to 4.
		fs (float, optional): Font scale for text. Defaults to 0.6.
	Returns:
		np.ndarray: The image with gaze results drawn on it.
	"""
	# Create canvas on which to draw predictions
	img_h, img_w, img_c = image.shape
	canvas = image.copy()

	# Scale of the drawing according to image resolution
	scale = max(img_h, img_w) / 1920
	fs *= scale
	thickness = int(scale * thickness)
	gaze_pt_size = int(scale * gaze_pt_size)
	head_center_size = int(scale * head_center_size)

	# Draw heatmap
	if heatmap_pid is not None:
		if len(gaze_heatmaps) == 0:
			raise ValueError("gaze_heatmaps must be provided if heatmap_pid is provided.")
		mask = (pids == heatmap_pid)
		if mask.sum() == 1: # only if detection found
			gaze_heatmap = gaze_heatmaps[mask]
			heatmap = TF.resize(gaze_heatmap, (img_h, img_w), antialias=True).squeeze().numpy()
			heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
			heatmap = cm.inferno(heatmap) * 255
			canvas = ((1 - alpha) * image + alpha * heatmap[..., :3]).astype(np.uint8)

			# Write pid being used for the heatmap
			hm_pid_text = f"Heatmap PID: {heatmap_pid}"
			(w_text, h_text), _ = cv2.getTextSize(hm_pid_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
			ul = (img_w - w_text - 20, img_h - h_text - 15)
			br = (img_w, img_h)
			cv2.rectangle(canvas, ul, br, (0, 0, 0), -1)
			hm_pid_text_loc = (img_w - w_text - 10, img_h - 10)
			cv2.putText(canvas, hm_pid_text, hm_pid_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

	# Draw head bboxes
	if len(head_bboxes) > 0:
		if len(pids) == 0:
			raise ValueError("pids must be provided if head_bboxes is provided.")

		# Convert to numpy
		head_bboxes = head_bboxes.numpy() if isinstance(head_bboxes, torch.Tensor) else np.array(head_bboxes)
		inouts = inouts.numpy() if isinstance(inouts, torch.Tensor) else np.array(inouts)
		if head_bboxes.max() <= 1.0:
			head_bboxes = head_bboxes * np.array([img_w, img_h, img_w, img_h])
		head_bboxes = head_bboxes.astype(int)

		# Compute head center
		head_centers = np.hstack([(head_bboxes[:,[0]] + head_bboxes[:,[2]]) / 2, (head_bboxes[:,[1]] + head_bboxes[:,[3]]) / 2])
		head_centers = head_centers.astype(int)

		gaze_available = (len(gaze_points) > 0)
		if gaze_available and (len(inouts) == 0):
			raise ValueError("inouts must be provided if gaze_pts is provided.")

		if gaze_available:
			gaze_points = gaze_points.numpy() if isinstance(gaze_points, torch.Tensor) else np.array(gaze_points)
			if (gaze_points.max() <= 1.):
				gaze_points = gaze_points * np.array([img_w, img_h])
			gaze_points = gaze_points.astype(int)

		if gaze_vecs is not None:
			gaze_vecs = gaze_vecs.numpy() if isinstance(gaze_vecs, torch.Tensor) else np.array(gaze_vecs)

		for i, head_bbox in enumerate(head_bboxes):
			pid = pids[i]
			if (heatmap_pid is not None) and (heatmap_pid != pid):
				continue

			xmin, ymin, xmax, ymax = head_bbox
			head_radius = max(xmax-xmin, ymax-ymin) // 2
			color = colors[pid % len(colors)]

			# Compute Head Center
			head_center = head_centers[i]

			head_bbox_ul = (xmin, ymin)
			head_bbox_br = (xmax, ymax)
			head_center_ul = head_center - (head_center_size // 2)
			head_center_br = head_center + (head_center_size // 2)
			cv2.rectangle(canvas, head_center_ul, head_center_br, color, -1) # head center point
			cv2.circle(canvas, head_center, head_radius, color, thickness) # head circle

			# Draw header
			io = inouts[i] if inouts is not None else "-"
			header_text = f"P{pid}: {io:.2f}"
			(w_text, h_text), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)

			header_ul =  (int(head_center[0] - w_text / 2), int(ymin - thickness / 2))
			header_br = (int(head_center[0] + w_text / 2), int(ymin + h_text + 5))
			cv2.rectangle(canvas, header_ul, header_br, color, -1) # header bbox
			cv2.putText(canvas, header_text, (header_ul[0], int(ymin + h_text)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA) # header text

			if gaze_available and (io > io_thr or not filter_by_inout):
				gp = gaze_points[i]
				vec = (gp - head_center)
				vec = vec / (np.linalg.norm(vec) + 0.000001)
				intersection = head_center + (vec * head_radius).astype(int)
				cv2.line(canvas, intersection, gp, color, thickness)

				cv2.circle(canvas, gp, gaze_pt_size, color, -1)

			if gaze_vecs is not None:
				gv = gaze_vecs[i]
				cv2.arrowedLine(canvas, head_center, (head_center + gaze_vec_factor * head_radius * gv).astype(int), color, thickness)


	# Write frame number
	if frame_nb is not None:
		frame_nb = str(frame_nb)
		(w_text, h_text), _ = cv2.getTextSize(frame_nb, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
		nb_ul = (int((img_w - w_text) / 2), (img_h - h_text - 15))
		nb_br = (int((img_w + w_text) / 2), img_h)
		cv2.rectangle(canvas, nb_ul, nb_br, (0, 0, 0), -1)
		nb_text_loc = (int((img_w - w_text) / 2), (img_h - 10))
		cv2.putText(canvas, frame_nb, nb_text_loc, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

	return canvas


def main():

	start = dt.datetime.now()

	# Path magic
	video_file = os.path.join(args.input_dir, args.input_filename)
	basename, ext = os.path.splitext(args.input_filename)
	csv_file = os.path.join(args.output_dir, f"{basename}-gaze.csv")

	if args.heatmap_pid >= 0:
		output_file = os.path.join(args.output_dir, f"{basename}-pid{args.heatmap_pid}-pred{ext}")
	else:
		output_file = os.path.join(args.output_dir, f"{basename}-pred{ext}")
	print(colored(f"Processing {video_file}", TERM_COLOR))

	# Load models
	tracker = load_tracker()
	head_detector = load_head_detection_model(DEVICE)
	sharingan = load_sharingan_model(CKPT_PATH, DEVICE)
	print(colored(f"Loaded tracker, head detector, and sharingan models.", TERM_COLOR))

	# Read Video Clip
	cap = cv2.VideoCapture(video_file)
	ret, frame = cap.read()
	img_h, img_w, _ = frame.shape  # retrieve video height and width
	fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Initialize ffmpeg writer (only when --write-video is set)
	process = None
	if args.write_video:
		ffmpeg_bin = get_ffmpeg_path()
		command = [
			ffmpeg_bin,
			"-loglevel", "error",
			"-y",
			"-s", f"{img_w}x{img_h}",
			"-pixel_format", "rgb24",
			"-f", "rawvideo",
			"-r", str(fps),
			"-i", "pipe:",
			"-vcodec", "libx264",
			"-pix_fmt", "yuv420p",
			"-crf", "24",
			output_file,
		]
		process = sp.Popen(command, stdin=sp.PIPE)

	# Open CSV for gaze-target output
	os.makedirs(args.output_dir, exist_ok=True)
	csv_fields = ["frame", "timestamp_sec", "looker_pid", "target_pid"]
	csv_fh = open(csv_file, "w", newline="")
	csv_writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
	csv_writer.writeheader()

	seen_pids = set()
	triad_warned = False

	# Iterate over frames and process
	frame_nb = 0
	with tqdm(total=frame_count) as pbar:
		while ret:
			frame_nb += 1
			timestamp_sec = round(frame_nb / fps, 4)

			# =============== Predict =============== #
			frame_np = frame[..., ::-1] # BGR >> RGB
			frame_pil = Image.fromarray(frame_np)
			output = predict_gaze(frame_pil, sharingan, head_detector, tracker=tracker)
			gaze_points, gaze_vecs, inouts, head_bboxes, gaze_heatmaps, pids = output

			num_people = len(head_bboxes)
			pids = np.arange(num_people) if len(pids) == 0 else pids

			# =============== Triad check =============== #
			frame_pids = set(int(p) for p in pids)
			seen_pids.update(frame_pids)
			if len(frame_pids) != 3 and not triad_warned:
				print(colored(f"WARNING: frame {frame_nb} has {len(frame_pids)} person(s) "
							  f"(expected 3). PIDs: {frame_pids}", "red"))
				triad_warned = True

			# =============== CSV rows =============== #
			gaze_rows = resolve_gaze_targets(
				pids, head_bboxes, gaze_points, inouts, img_w, img_h,
				io_thr=args.io_thr)
			for row in gaze_rows:
				row["frame"] = frame_nb
				row["timestamp_sec"] = timestamp_sec
				csv_writer.writerow(row)

			# =============== Draw Prediction (optional) =============== #
			if args.write_video:
				heatmap_pid = args.heatmap_pid if args.heatmap_pid >= 0 else None
				frame_draw = draw_gaze(frame_np,
								head_bboxes = head_bboxes,
								gaze_points = gaze_points,
								gaze_vecs = gaze_vecs if args.show_gaze_vec else None,
								inouts = inouts,
								pids = pids,
								gaze_heatmaps = gaze_heatmaps,
								heatmap_pid = heatmap_pid,
								frame_nb = None,
								colors = COLORS,
								filter_by_inout = args.filter_by_inout,
								alpha = 0.6,
								gaze_pt_size = 20,
								gaze_vec_factor = 0.6,
								head_center_size = 18,
								thickness = 10,
								fs = 0.8,
								)
				process.stdin.write(frame_draw.tobytes())

			# =============== Read Next Frame =============== #
			ret, frame = cap.read()

			pbar.update(1)

	# Release Capture Device
	cap.release()

	# Close CSV
	csv_fh.close()
	print(colored(f"CSV saved to {csv_file}", TERM_COLOR))

	# Triad summary warning
	if len(seen_pids) != 3:
		print(colored(f"WARNING: expected 3 unique participants across the video, "
					  f"but found {len(seen_pids)}: {seen_pids}", "red"))

	# Close ffmpeg
	if process is not None:
		process.stdin.close()
		process.wait()
		process.terminate()
		print(colored(f"Video saved to {output_file}", TERM_COLOR))

	end = dt.datetime.now()
	print(colored(f"Finished. The script took {end - start}.", TERM_COLOR))


if __name__ == "__main__":
	main()
