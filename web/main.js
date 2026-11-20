import { HeadTracker } from "./tracker.js";            // NEW

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const W = canvas.width, H = canvas.height;
const HEAD_R = 62;                                       // NEW (moves to physics.js later)

let tracker = null;                                      // NEW

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function drawFrame() {
  ctx.save();
  ctx.translate(W, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();
}

function drawText(text, x, y, color, size = 30) {      // NEW (like draw_text in Python)
  ctx.font = `bold ${size}px Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.fillStyle = "black";
  ctx.fillText(text, x + 2, y + 2);                      // shadow
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function loop() {
  drawFrame();

  const head = tracker.update(video, W, H);             // NEW
  if (head) {
    ctx.strokeStyle = "rgb(60, 220, 120)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    drawText("FACE NOT DETECTED", W / 2, 70, "rgb(240, 80, 80)");
  }

  requestAnimationFrame(loop);
}

async function start() {                                // NEW
  await startCamera();
  drawFrame();
  drawText("LOADING FACE TRACKER...", W / 2, H / 2, "orange");
  tracker = await HeadTracker.create();
  loop();
}

start().catch((err) => {
  drawText("Error: " + err.message, W / 2, H / 2, "red");
});