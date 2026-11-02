"""
data_logger.py — records gameplay for imitation learning, one CSV row per frame.

Every position/velocity is saved NORMALISED by screen size (x/W, y/H, vx/W, vy/H),
so sessions recorded at 1280x720 and 640x480 can be mixed in one dataset.

Files go to data/raw/session_YYYYmmdd_HHMMSS.csv (one file per game run).
"""

import csv
import os
import time
from datetime import datetime

FIELDS = [
    "t", "dt",                                   # seconds since start, frame time
    "ball_x", "ball_y", "ball_vx", "ball_vy",    # ball state
    "head_x", "head_y", "head_vx", "head_vy",    # YOUR head (the thing we'll imitate)
    "bot_x", "bot_y",                            # partner position (where to pass)
    "human_hit", "bot_hit", "dropped",           # events this frame (0/1)
    "rally",                                     # rally count at this frame
]


class GameLogger:
    def __init__(self, W, H, folder="data/raw"):
        os.makedirs(folder, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(folder, f"session_{stamp}.csv")
        self.W, self.H = W, H
        self.file = open(self.path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()
        self.t0 = time.perf_counter()
        self.rows = 0
        self.enabled = True          # press L in game to pause/resume recording

    def log(self, dt, ball, head, head_vel, bot_pos,
            human_hit, bot_hit, dropped, rally):
        if not self.enabled:
            return
        W, H = self.W, self.H
        self.writer.writerow({
            "t": round(time.perf_counter() - self.t0, 4),
            "dt": round(dt, 4),
            "ball_x": round(ball.x / W, 5),
            "ball_y": round(ball.y / H, 5),
            "ball_vx": round(ball.vx / W, 5),
            "ball_vy": round(ball.vy / H, 5),
            "head_x": round(head[0] / W, 5),
            "head_y": round(head[1] / H, 5),
            "head_vx": round(head_vel[0] / W, 5),
            "head_vy": round(head_vel[1] / H, 5),
            "bot_x": round(bot_pos[0] / W, 5),
            "bot_y": round(bot_pos[1] / H, 5),
            "human_hit": int(human_hit),
            "bot_hit": int(bot_hit),
            "dropped": int(dropped),
            "rally": rally,
        })
        self.rows += 1
        if self.rows % 300 == 0:     # flush every ~10 s so a crash loses little
            self.file.flush()

    def close(self):
        self.file.close()
        if self.rows == 0:           # don't leave empty files around
            os.remove(self.path)
        return self.path, self.rows