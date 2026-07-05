#!/usr/bin/env python3
import os, sys, json, time, base64, argparse, threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import requests

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ENDPOINT = "aHR0cHM6Ly9hcmsuY24tYmVpamluZy52b2xjZXMuY29tL2FwaS92My9jb250ZW50cy9nZW5lcmF0aW9ucy90YXNrcw=="
_CREDENTIAL = "YXJrLWE2ZTA5NmYwLTg5ZTgtNDJmNS04YTY0LTNlOTczOTU1NTliNS05M2I2Mg=="
_MODEL = "ZG91YmFvLXNlZWRhbmNlLTItMC0yNjAxMjg="
_MODEL_FAST = "ZG91YmFvLXNlZWRhbmNlLTItMC1mYXN0LTI2MDEyOA=="

def _d(s):
    return base64.b64decode(s).decode("utf-8")

ACCIDENT_PROMPTS = {
    "animal_intrusion": (
        "Dashcam POV: an autonomous off-road SUV driving on a narrow dirt mountain trail "
        "surrounded by dense green bushes. Suddenly, a large deer bursts out from the right-side "
        "vegetation, darting directly into the vehicle's path within 15 meters. "
        "The animal moves fast from right to left across the road. Leaves and dust scatter. "
        "The deer fills the camera frame rapidly. Photorealistic, raw dashcam style, "
        "natural lighting, no text overlay, no HUD elements, smooth motion."
    ),
    "rockfall": (
        "Front-facing vehicle camera: an off-road autonomous SUV traversing a rocky mountain trail "
        "with a steep cliff on the left side. Multiple rocks and boulders suddenly break loose "
        "from the slope above, tumbling and bouncing down toward the road. Rocks of various sizes "
        "cascade down, some as large as basketballs. Dust clouds erupt from rock impacts. "
        "Small gravel sprays across the road surface. The rocks roll into the vehicle's path "
        "within 2-3 seconds. Photorealistic, dashcam aesthetic, natural mountain lighting, "
        "realistic rock physics, no text or HUD."
    ),
    "terrain_collapse": (
        "Dashcam view: an off-road autonomous SUV driving on a dirt road along a riverbank. "
        "The road surface ahead suddenly cracks and collapses, forming a growing sinkhole "
        "that expands from 1 meter to 5 meters wide within seconds. Soil and rocks crumble "
        "into the cavity. The vehicle approaches the collapsing edge, front wheels near the "
        "newly formed precipice. Water visible in the hole below. Photorealistic, raw camera "
        "style, natural daylight, realistic soil deformation, no text overlay."
    ),
    "sudden_skidding": (
        "Hood-mounted camera POV: an off-road autonomous SUV driving at moderate speed on a "
        "loose gravel mountain trail. The vehicle suddenly hits an extremely low-friction patch "
        "of mud and loose stones. Rear wheels lose traction instantly, the vehicle begins "
        "sliding and fishtailing sideways. Dust and mud spray from spinning tires. "
        "Camera view tilts and shakes as vehicle loses stability. Visible skid marks on gravel. "
        "Dust particles in sunlight. Photorealistic, raw dashcam aesthetic, realistic vehicle "
        "dynamics, motion blur on wheels, no text or HUD elements."
    ),
}

NEGATIVE_PROMPT = (
    "text, watermark, logo, HUD overlay, speedometer, cartoon, anime, 3D render, "
    "low quality, blurry, pixelated, city street, paved highway, traffic lights, "
    "indoor, underwater, unrealistic physics"
)

class VideoGenerator:
    def __init__(self):
        endpoint = _d(_ENDPOINT)
        credential = _d(_CREDENTIAL)
        self.tasks_url = endpoint
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        })

    def _submit(self, prompt, resolution="720p", ratio="16:9", duration=5,
                image_paths=None, video_paths=None, audio_paths=None, fast=False):
        model = _d(_MODEL_FAST if fast else _MODEL)
        content_parts = [{"type": "text", "text": prompt}]
        if image_paths:
            for p in image_paths:
                if os.path.exists(p):
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": self._b64_image(p)},
                        "role": "reference_image",
                    })
        if video_paths:
            for p in video_paths:
                if os.path.exists(p):
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {"url": self._b64_video(p)},
                        "role": "reference_video",
                    })
                elif p.startswith("http"):
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {"url": p},
                        "role": "reference_video",
                    })
        if audio_paths:
            for p in audio_paths:
                if os.path.exists(p):
                    content_parts.append({
                        "type": "audio_url",
                        "audio_url": {"url": self._b64_audio(p)},
                        "role": "reference_audio",
                    })
                elif p.startswith("http"):
                    content_parts.append({
                        "type": "audio_url",
                        "audio_url": {"url": p},
                        "role": "reference_audio",
                    })
        payload = {
            "model": model, "content": content_parts,
            "ratio": ratio, "duration": duration,
            "watermark": False, "generate_audio": False,
        }
        if resolution:
            payload["resolution"] = resolution
        print(f"[INFO] Submitting generation task...")
        print(f"[INFO] Resolution: {resolution}, Duration: {duration}s, Ratio: {ratio}")
        r = self.session.post(self.tasks_url, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Submission failed ({r.status_code}): {r.text[:500]}")
        data = r.json()
        tid = data.get("id", "")
        print(f"[INFO] Task submitted: {tid}")
        return tid

    def _poll(self, task_id, max_wait=600.0):
        print(f"[INFO] Waiting for completion (max {max_wait:.0f}s)...")
        start = time.time()
        wait = 10.0
        spinner = ["|", "/", "-", "\\"]
        si = 0
        while True:
            elapsed = time.time() - start
            if elapsed > max_wait:
                raise TimeoutError(f"Task timed out ({max_wait}s)")
            r = self.session.get(f"{self.tasks_url}/{task_id}", timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"Status check failed: {r.text[:300]}")
            result = r.json()
            status = result.get("status", "unknown")
            m, s = int(elapsed // 60), int(elapsed % 60)
            print(f"\r  {spinner[si % 4]} [{m:02d}:{s:02d}] {status}", end="", flush=True)
            si += 1
            if status == "succeeded":
                print(f"\r  OK [{m:02d}:{s:02d}] completed")
                return result
            elif status in ("failed", "expired", "cancelled"):
                err = result.get("error", {}).get("message", "unknown")
                print(f"\r  FAIL [{m:02d}:{s:02d}]: {err}")
                raise RuntimeError(f"Task failed ({status}): {err}")
            time.sleep(wait)
            wait = min(wait * 1.5, 60)

    def _download(self, url, path):
        print(f"[INFO] Downloading...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                done += len(chunk)
                if total > 0:
                    pct = done * 100 // total
                    print(f"\r  {done/(1024*1024):.1f}/{total/(1024*1024):.1f} MB ({pct}%)", end="", flush=True)
        print(f"\r  Downloaded: {path} ({done/(1024*1024):.1f} MB)")
        return path

    def generate(self, prompt, output_path=None, resolution="720p", ratio="16:9",
                 duration=5, image_paths=None, video_paths=None, audio_paths=None, fast=False):
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./outputs/generated_{ts}.mp4"
        if image_paths is None:
            image_paths = []
        tid = self._submit(prompt=prompt, resolution=resolution, ratio=ratio,
                           duration=duration, image_paths=image_paths,
                           video_paths=video_paths, audio_paths=audio_paths, fast=fast)
        result = self._poll(tid)
        content = result.get("content", {})
        video_url = content.get("video_url", "") or result.get("video_url", "")
        if not video_url:
            raise RuntimeError("No video URL in response")
        return self._download(video_url, output_path)

    @staticmethod
    def _b64_image(path):
        with open(path, "rb") as f:
            d = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lower().lstrip(".")
        m = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "bmp": "bmp"}
        return f"data:image/{m.get(ext, 'jpeg')};base64,{d}"

    @staticmethod
    def _b64_video(path):
        with open(path, "rb") as f:
            d = base64.b64encode(f.read()).decode()
        return f"data:video/mp4;base64,{d}"

    @staticmethod
    def _b64_audio(path):
        with open(path, "rb") as f:
            d = base64.b64encode(f.read()).decode()
        ext = Path(path).suffix.lower().lstrip(".")
        m = {"mp3": "mpeg", "wav": "wav", "ogg": "ogg", "m4a": "mp4"}
        return f"data:audio/{m.get(ext, 'mpeg')};base64,{d}"


def play_video_window(video_path, title="Generated Video", accident_type=""):
    import numpy as np, imageio
    from PIL import Image, ImageTk
    import tkinter as tk
    from tkinter import ttk

    print(f"\n[INFO] Opening video player...")
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get("fps", 24.0)
        frames = []
        for frame in reader:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            elif frame.shape[0] == 3 and frame.ndim == 3:
                frame = frame.transpose(1, 2, 0)
            frames.append(frame)
        reader.close()
    except Exception as e:
        print(f"[WARN] Cannot read video: {e}")
        os.startfile(video_path)
        return

    if not frames:
        os.startfile(video_path)
        return

    total_frames = len(frames)
    COLORS = {"animal_intrusion": "#FF8C00", "rockfall": "#808080",
              "terrain_collapse": "#8B4513", "sudden_skidding": "#DC143C"}
    accent = COLORS.get(accident_type, "#4a90d9")

    root = tk.Tk()
    root.title(title)
    root.configure(bg="#0d0d0d")
    h, w = frames[0].shape[:2]
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    scale = min(1.0, sw * 0.85 / w, sh * 0.78 / h)
    dw, dh = int(w * scale), int(h * scale)
    root.geometry(f"{dw + 30}x{dh + 110}+{sw//2 - dw//2}+{sh//2 - dh//2}")
    root.minsize(500, 350)

    top_bar = tk.Frame(root, bg="#141414", height=36)
    top_bar.pack(fill=tk.X, padx=8, pady=(8, 0))
    top_bar.pack_propagate(False)
    if accident_type:
        tk.Label(top_bar, text=f" {accident_type.upper().replace('_', ' ')} ",
                 bg=accent, fg="white", font=("Microsoft YaHei UI", 10, "bold"),
                 padx=10, pady=2).pack(side=tk.LEFT, padx=(0, 10))
    tk.Label(top_bar, text=f"Off-Road Accident Video Generator",
             bg="#141414", fg="#999999", font=("Microsoft YaHei UI", 9)).pack(side=tk.LEFT)
    fps_lbl = tk.Label(top_bar, text=f"{fps:.0f} FPS  |", bg="#141414", fg="#777777",
                       font=("Consolas", 9))
    fps_lbl.pack(side=tk.RIGHT)
    frame_lbl = tk.Label(top_bar, text="1 / 1", bg="#141414", fg="#aaaaaa",
                         font=("Consolas", 10))
    frame_lbl.pack(side=tk.RIGHT, padx=(0, 5))

    cf = tk.Frame(root, bg="#000000")
    cf.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
    canvas = tk.Canvas(cf, width=dw, height=dh, bg="#000000", highlightthickness=0, cursor="hand2")
    canvas.pack(expand=True)

    ctrl = tk.Frame(root, bg="#141414", height=50)
    ctrl.pack(fill=tk.X, padx=8, pady=(0, 8))
    ctrl.pack_propagate(False)

    state = {"playing": False, "current": 0, "stop": threading.Event()}
    photo_ref = [None]
    progress_var = tk.DoubleVar(value=0)

    def show_frame(idx):
        idx = max(0, min(idx, total_frames - 1))
        state["current"] = idx
        img = Image.fromarray(frames[idx]).resize((dw, dh), Image.LANCZOS)
        photo_ref[0] = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(dw // 2, dh // 2, image=photo_ref[0], anchor=tk.CENTER)
        frame_lbl.config(text=f"{idx + 1} / {total_frames}  |")
        progress_var.set((idx + 1) / total_frames * 100)

    def play_loop():
        while not state["stop"].is_set():
            if state["playing"] and state["current"] < total_frames - 1:
                state["current"] += 1
                root.after(0, lambda i=state["current"]: show_frame(i))
                delay = 1.0 / max(fps, 1.0)
            elif state["current"] >= total_frames - 1:
                state["playing"] = False
                root.after(0, lambda: btn_play.config(text="Replay"))
                delay = 0.1
            else:
                delay = 0.05
            time.sleep(delay)

    def toggle():
        if state["playing"]:
            state["playing"] = False
            btn_play.config(text="Play")
        else:
            if state["current"] >= total_frames - 1:
                state["current"] = 0; show_frame(0)
            state["playing"] = True
            btn_play.config(text="Pause")

    def close():
        state["stop"].set(); state["playing"] = False; root.destroy()

    original_fps = fps

    btn_play = tk.Button(ctrl, text="Play", command=toggle, bg="#1a1a1a", fg="white",
                         font=("Microsoft YaHei UI", 10), relief=tk.FLAT, padx=12, pady=4,
                         activebackground="#333333", cursor="hand2")
    btn_play.pack(side=tk.LEFT, padx=(6, 10))

    pf = tk.Frame(ctrl, bg="#141414")
    pf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    pc = tk.Canvas(pf, height=30, bg="#141414", highlightthickness=0)
    pc.pack(fill=tk.X)

    def draw_progress(*_):
        pc.delete("all")
        cw = pc.winfo_width()
        pct = progress_var.get() / 100
        pc.create_rectangle(0, 11, cw, 19, fill="#2a2a2a", outline="")
        if pct > 0:
            pc.create_rectangle(0, 11, cw * pct, 19, fill=accent, outline="")
        pc.create_oval(cw * pct - 7, 7, cw * pct + 7, 23, fill=accent, outline="")

    def on_click(event):
        pct = event.x / pc.winfo_width()
        target = int(pct * (total_frames - 1))
        was = state["playing"]; state["playing"] = False; show_frame(target)
        if was: state["playing"] = True

    pc.bind("<Button-1>", on_click)
    pc.bind("<B1-Motion>", on_click)
    progress_var.trace_add("write", draw_progress)
    pc.bind("<Configure>", draw_progress)

    sv = tk.StringVar(value="1x")
    sc = ttk.Combobox(ctrl, textvariable=sv, values=["0.5x", "1x", "2x", "3x"],
                      width=5, state="readonly")
    sc.pack(side=tk.RIGHT, padx=(5, 0))
    sc.bind("<<ComboboxSelected>>", lambda e: (
        exec(f"global fps; fps = {original_fps} * {sv.get().replace('x','')}"),
        fps_lbl.config(text=f"{fps:.0f} FPS  |")))

    tk.Label(ctrl, text="Speed:", bg="#141414", fg="#777",
             font=("Microsoft YaHei UI", 8)).pack(side=tk.RIGHT)

    tk.Button(ctrl, text="Close", command=close, bg="#2a2a2a", fg="#dddddd",
              font=("Microsoft YaHei UI", 10), relief=tk.FLAT, padx=14, pady=4,
              activebackground="#444444", cursor="hand2").pack(side=tk.RIGHT, padx=(15, 8))

    root.bind("<space>", lambda e: toggle())
    root.bind("<Escape>", lambda e: close())
    root.bind("<f>", lambda e: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
    root.bind("<Left>", lambda e: (setattr(state, 'playing', False), show_frame(state["current"] - 3)))
    root.bind("<Right>", lambda e: (setattr(state, 'playing', False), show_frame(state["current"] + 3)))
    root.protocol("WM_DELETE_WINDOW", close)

    show_frame(0)
    threading.Thread(target=play_loop, daemon=True).start()
    root.after(300, toggle)
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()
    print("[INFO] Player closed")


def main():
    parser = argparse.ArgumentParser(description="Off-Road Accident Video Generator")
    parser.add_argument("--accident", "-a", type=str,
                        choices=["animal_intrusion", "rockfall", "terrain_collapse", "sudden_skidding"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--prompt", "-p", type=str, default=None)
    parser.add_argument("--image", "-i", type=str, default=None)
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--resolution", "-r", type=str, default="720p",
                        choices=["480p", "720p", "1080p"])
    parser.add_argument("--duration", "-d", type=int, default=5)
    parser.add_argument("--ratio", type=str, default="16:9", choices=["16:9", "9:16", "1:1"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--no-play", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./outputs")

    args = parser.parse_args()

    if not args.accident and not args.prompt and not args.all:
        print("[ERROR] Specify --accident, --prompt, or --all")
        sys.exit(1)

    print("=" * 60)
    print("  Off-Road Accident Video Generator")
    print("=" * 60)

    gen = VideoGenerator()
    to_generate = []
    if args.all:
        to_generate = ["animal_intrusion", "rockfall", "terrain_collapse", "sudden_skidding"]
    elif args.accident:
        to_generate = [args.accident]
    elif args.prompt:
        to_generate = ["custom"]

    results = []
    for atype in to_generate:
        print(f"\n{'─' * 60}")
        print(f"  {atype.upper().replace('_', ' ')}")
        print(f"{'─' * 60}")
        if args.prompt:
            prompt = args.prompt
        elif atype == "custom":
            continue
        else:
            prompt = ACCIDENT_PROMPTS.get(atype, "")
        print(f"  Prompt: {prompt[:120]}...")

        if args.output and not args.all:
            out = args.output
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(args.output_dir, f"{atype}_{ts}.mp4")

        imgs = args.images if args.images else ([args.image] if args.image else [])
        vids = [args.video] if args.video else []
        auds = [args.audio] if args.audio else []

        try:
            vp = gen.generate(prompt=prompt, output_path=out, resolution=args.resolution,
                              ratio=args.ratio, duration=args.duration,
                              image_paths=imgs if imgs else None,
                              video_paths=vids if vids else None,
                              audio_paths=auds if auds else None, fast=args.fast)
            results.append((atype, vp))
        except Exception as e:
            print(f"\n[ERROR] Failed ({atype}): {e}")

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(results)}/{len(to_generate)} videos")
    print(f"{'=' * 60}")
    for atype, path in results:
        sz = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        print(f"  OK {atype}: {path} ({sz:.1f} MB)")
    print("=" * 60)

    if not args.no_play and results:
        for atype, path in results:
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                play_video_window(path, title=f"Accident: {atype.replace('_', ' ').title()}",
                                  accident_type=atype)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
