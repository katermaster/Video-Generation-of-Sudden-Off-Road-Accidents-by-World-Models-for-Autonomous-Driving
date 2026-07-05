"""
Video Player Window — displays generated accident videos in a new tkinter window.

Features:
  - Play/pause toggle (Space key or button)
  - Seek bar with drag support
  - Frame counter display
  - Accident type overlay
  - Auto-close on video end (configurable)
  - Fullscreen toggle (F key)
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Callable

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import imageio
import numpy as np


class VideoPlayer:
    """
    A tkinter-based video player window for displaying generated accident videos.

    Usage:
        player = VideoPlayer("path/to/video.mp4", title="Sudden Skidding Accident")
        player.play()  # Blocks until window is closed

        # Or non-blocking:
        player.play(blocking=False)
    """

    def __init__(
        self,
        video_path: str,
        title: str = "Accident Video Player",
        accident_type: str = "",
        fps: Optional[float] = None,
        auto_close: bool = False,
        on_close: Optional[Callable] = None,
    ):
        self.video_path = video_path
        self.title = title
        self.accident_type = accident_type
        self.auto_close = auto_close
        self.on_close = on_close

        # Video state
        self.frames: list = []
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.is_playing: bool = False
        self.video_fps: float = fps or 10.0
        self.frame_delay: float = 1.0 / self.video_fps

        # Threading
        self._play_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # GUI state
        self.root: Optional[tk.Tk] = None
        self.canvas: Optional[tk.Canvas] = None
        self.label_frame: Optional[ttk.Label] = None
        self.seek_var: Optional[tk.DoubleVar] = None
        self.seek_bar: Optional[ttk.Scale] = None
        self.btn_play: Optional[ttk.Button] = None
        self.photo: Optional[ImageTk.PhotoImage] = None

        # Colors
        self.COLORS = {
            "animal_intrusion": "#FF8C00",  # Dark orange
            "rockfall": "#808080",          # Gray
            "terrain_collapse": "#8B4513",  # Saddle brown
            "sudden_skidding": "#DC143C",   # Crimson
        }

        # Load video
        self._load_video()

    # ============================================================
    # Video Loading
    # ============================================================

    def _load_video(self):
        """Load video frames from file."""
        try:
            reader = imageio.get_reader(self.video_path)
            meta = reader.get_meta_data()

            if self.video_fps is None:
                self.video_fps = meta.get("fps", 10.0)
            self.frame_delay = 1.0 / max(self.video_fps, 1.0)

            self.frames = []
            for frame in reader:
                # Convert to RGB if needed
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.shape[-1] == 4:
                    frame = frame[:, :, :3]
                elif frame.shape[-1] == 3 and frame.shape[0] == 3:
                    # CHW format
                    frame = frame.transpose(1, 2, 0)

                self.frames.append(frame)

            reader.close()
            self.total_frames = len(self.frames)

        except Exception as e:
            print(f"[ERROR] Failed to load video: {e}")
            # Create a placeholder frame
            self.frames = [np.zeros((540, 960, 3), dtype=np.uint8) + 60]
            self.total_frames = 1

        print(f"[INFO] Loaded {self.total_frames} frames @ {self.video_fps:.1f} fps")

    # ============================================================
    # GUI Setup
    # ============================================================

    def _setup_gui(self):
        """Build the tkinter window."""
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.configure(bg="#1a1a1a")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Set minimum size
        self.root.minsize(640, 400)

        # Get first frame for sizing
        first_frame = self.frames[0]
        h, w = first_frame.shape[:2]

        # Scale down if too large for screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        scale = min(1.0, screen_w * 0.85 / w, screen_h * 0.75 / h)
        self.display_w = int(w * scale)
        self.display_h = int(h * scale)

        self.root.geometry(f"{self.display_w + 40}x{self.display_h + 120}")

        # === Top frame: info bar ===
        top_frame = tk.Frame(self.root, bg="#1a1a1a", height=30)
        top_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

        # Accident type badge
        accent_color = self.COLORS.get(self.accident_type, "#4a90d9")
        if self.accident_type:
            badge = tk.Label(
                top_frame,
                text=f" {self.accident_type.upper().replace('_', ' ')} ",
                bg=accent_color,
                fg="white",
                font=("Segoe UI", 10, "bold"),
                padx=8, pady=2,
            )
            badge.pack(side=tk.LEFT)

        # Video title
        video_name = Path(self.video_path).name
        title_label = tk.Label(
            top_frame,
            text=f"  {video_name}",
            bg="#1a1a1a", fg="#aaaaaa",
            font=("Segoe UI", 9),
        )
        title_label.pack(side=tk.LEFT)

        # Frame counter
        self.label_frame = tk.Label(
            top_frame,
            text="0 / 0",
            bg="#1a1a1a", fg="#888888",
            font=("Segoe UI", 9),
        )
        self.label_frame.pack(side=tk.RIGHT)

        # === Center: video canvas ===
        canvas_frame = tk.Frame(self.root, bg="#000000")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.display_w,
            height=self.display_h,
            bg="#000000",
            highlightthickness=0,
        )
        self.canvas.pack(expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # === Bottom: controls ===
        ctrl_frame = tk.Frame(self.root, bg="#1a1a1a")
        ctrl_frame.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Play/Pause button
        self.btn_play = ttk.Button(
            ctrl_frame, text="▶  Play",
            command=self._toggle_playback,
        )
        self.btn_play.pack(side=tk.LEFT, padx=(0, 10))

        # Seek bar
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_bar = ttk.Scale(
            ctrl_frame,
            from_=0, to=max(1, self.total_frames - 1),
            variable=self.seek_var,
            orient=tk.HORIZONTAL,
            command=self._on_seek,
        )
        self.seek_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Speed control
        speed_frame = tk.Frame(ctrl_frame, bg="#1a1a1a")
        speed_frame.pack(side=tk.RIGHT, padx=(10, 0))
        tk.Label(speed_frame, text="Speed:", bg="#1a1a1a", fg="#888",
                 font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.speed_var = tk.StringVar(value="1x")
        speed_combo = ttk.Combobox(
            speed_frame, textvariable=self.speed_var,
            values=["0.5x", "1x", "2x", "3x"],
            width=5, state="readonly",
        )
        speed_combo.pack(side=tk.LEFT, padx=(3, 0))
        speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        # Close button
        close_btn = tk.Button(
            ctrl_frame, text="✕ Close", bg="#333333", fg="#ffffff",
            font=("Segoe UI", 9), relief=tk.FLAT, padx=10,
            command=self._on_close, activebackground="#555555",
        )
        close_btn.pack(side=tk.RIGHT, padx=(15, 0))

        # === Keyboard bindings ===
        self.root.bind("<space>", lambda e: self._toggle_playback())
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.bind("<f>", lambda e: self._toggle_fullscreen())
        self.root.bind("<F11>", lambda e: self._toggle_fullscreen())
        self.root.bind("<Left>", lambda e: self._seek_relative(-5))
        self.root.bind("<Right>", lambda e: self._seek_relative(5))

        # Show first frame
        self._show_frame(0)

    # ============================================================
    # Frame Display
    # ============================================================

    def _show_frame(self, idx: int):
        """Display frame at given index."""
        if not self.root or not self.canvas:
            return

        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx

        frame = self.frames[idx]

        # Convert to PIL Image
        img = Image.fromarray(frame)
        img = img.resize((self.display_w, self.display_h), Image.LANCZOS)

        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_w // 2, self.display_h // 2,
            image=self.photo, anchor=tk.CENTER,
        )

        # Update UI
        if self.label_frame:
            self.label_frame.config(text=f"{idx + 1} / {self.total_frames}")
        if self.seek_var:
            self.seek_var.set(idx)

    # ============================================================
    # Playback Control
    # ============================================================

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self.is_playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        """Start playback."""
        if self.is_playing:
            return

        self.is_playing = True
        self._stop_event.clear()
        self.btn_play.config(text="⏸  Pause")

        self._play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._play_thread.start()

    def _pause(self):
        """Pause playback."""
        self.is_playing = False
        self._stop_event.set()
        if self.btn_play:
            self.btn_play.config(text="▶  Play")

    def _playback_loop(self):
        """Main playback loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            # Show current frame
            if self.root and self.current_frame < self.total_frames:
                self.root.after(0, lambda i=self.current_frame: self._show_frame(i))
                self.current_frame += 1
            else:
                # Reached end
                self.root.after(0, self._on_video_end)
                break

            # Wait for next frame
            elapsed = 0
            while elapsed < self.frame_delay and not self._stop_event.is_set():
                time.sleep(0.01)
                elapsed += 0.01

    def _on_video_end(self):
        """Handle video end."""
        self._pause()
        if self.auto_close:
            self._on_close()
        else:
            # Loop back to start
            self.current_frame = 0
            self._show_frame(0)

    # ============================================================
    # UI Events
    # ============================================================

    def _on_canvas_click(self, event):
        """Click on video to toggle play/pause."""
        self._toggle_playback()

    def _on_seek(self, value):
        """Seek bar changed."""
        idx = int(float(value))
        if idx != self.current_frame:
            was_playing = self.is_playing
            if was_playing:
                self._pause()
            self._show_frame(idx)
            if was_playing:
                self._play()

    def _seek_relative(self, delta: int):
        """Seek by delta frames."""
        target = max(0, min(self.total_frames - 1, self.current_frame + delta))
        was_playing = self.is_playing
        if was_playing:
            self._pause()
        self._show_frame(target)
        if was_playing:
            self._play()

    def _on_speed_change(self, event=None):
        """Speed combo changed."""
        speed_str = self.speed_var.get().replace("x", "")
        try:
            speed = float(speed_str)
            self.frame_delay = 1.0 / (self.video_fps * speed)
        except ValueError:
            pass

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.root:
            is_fs = self.root.attributes("-fullscreen")
            self.root.attributes("-fullscreen", not is_fs)

    def _on_close(self):
        """Close the window."""
        self._pause()
        if self.root:
            self.root.destroy()
            self.root = None
        if self.on_close:
            self.on_close()

    # ============================================================
    # Public API
    # ============================================================

    def play(self, blocking: bool = True):
        """
        Open the video player window and start playback.

        Args:
            blocking: If True, blocks until window is closed.
                      If False, returns immediately (window runs in main thread).
        """
        self._setup_gui()
        self._show_frame(0)

        if blocking:
            self._play()
            # Bring window to front
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(100, lambda: self.root.attributes("-topmost", False))
            self.root.mainloop()
        else:
            self._play()
            # Schedule mainloop in background (not thread-safe but works for simple cases)
            threading.Thread(target=self.root.mainloop, daemon=True).start()


def play_video(
    video_path: str,
    title: str = "Generated Accident Video",
    accident_type: str = "",
    blocking: bool = True,
) -> None:
    """
    Quick function to open a video player window.

    Args:
        video_path: Path to .mp4 file
        title: Window title
        accident_type: One of "animal_intrusion", "rockfall", "terrain_collapse", "sudden_skidding"
        blocking: If True, blocks until window is closed
    """
    player = VideoPlayer(
        video_path=video_path,
        title=title,
        accident_type=accident_type,
    )
    player.play(blocking=blocking)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        video_file = sys.argv[1]
    else:
        # Try to find a generated video
        outputs_dir = Path(__file__).parent.parent / "outputs"
        mp4_files = list(outputs_dir.glob("*.mp4"))
        if mp4_files:
            video_file = str(mp4_files[0])
            print(f"[INFO] Found: {video_file}")
        else:
            # Create a simple test video
            print("[INFO] No video found, creating test pattern...")
            import imageio
            test_frames = []
            for i in range(30):
                frame = np.zeros((270, 480, 3), dtype=np.uint8)
                cv2_color = [
                    int(50 + 30 * np.sin(i * 0.3)),
                    int(100 + 50 * np.sin(i * 0.2)),
                    int(150 + 80 * np.sin(i * 0.15)),
                ]
                frame[:, :] = cv2_color
                # Moving circle
                cx = int(240 + 100 * np.sin(i * 0.2))
                cy = int(135 + 50 * np.cos(i * 0.15))
                for y in range(270):
                    for x in range(480):
                        if (x - cx) ** 2 + (y - cy) ** 2 < 400:
                            frame[y, x] = [255, 100, 50]
                test_frames.append(frame)

            test_path = str(Path(__file__).parent.parent / "outputs" / "test_pattern.mp4")
            os.makedirs(os.path.dirname(test_path), exist_ok=True)
            imageio.mimsave(test_path, test_frames, fps=10)
            video_file = test_path

    print(f"Playing: {video_file}")
    play_video(
        video_file,
        title="Accident Video Demo",
        accident_type="sudden_skidding",
        blocking=True,
    )
