import { HeadTracker } from "./tracker.js";
import { Ball, BALL_R, HEAD_R, MAX_DT, HIT_COOLDOWN, tryHeader } from "./physics.js";   // NEW

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const W = canvas.width, H = canvas.height;

let tracker = null;
let ball = null;                                               // NEW
let lastTime = 0;
let humanLastHit = 0;                                           // NEW

// like random.uniform(a, b) in Python                         // NEW
function uniform(a, b) {
  return a + Math.random() * (b - a);
}

// same as spawn_on_human_side() in pass_game.py               // NEW
function spawnOnHumanSide() {
  const b = new Ball(W * uniform(0.18, 0.32), H * 0.12);
  b.vx = uniform(-30, 30);
  return b;
}

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

function drawText(text, x, y, color, size = 30) {
  ctx.font = `bold ${size}px Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.fillStyle = "black";
  ctx.fillText(text, x + 2, y + 2);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function loop(time) {                                          // NEW: browser passes the time
  const dt = Math.min((time - lastTime) / 1000, MAX_DT);       // NEW: like clock.tick()
  lastTime = time;                                             // NEW

  drawFrame();
  const head = tracker.update(video, W, H);

  if (head) {
    ball.step(dt, W);       
    const now = performance.now() / 1000;
    if (now - humanLastHit > HIT_COOLDOWN && tryHeader(ball, head, tracker.velocity())) {
      humanLastHit = now;
    }                                   // NEW
    if (ball.y - BALL_R > H) ball = spawnOnHumanSide();        // NEW: fell off the bottom

    ctx.strokeStyle = "rgb(60, 220, 120)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    drawText("FACE NOT DETECTED", W / 2, 70, "rgb(240, 80, 80)");
  }

  ball.draw(ctx);                                              // NEW
  requestAnimationFrame(loop);
}

async function start() {
  await startCamera();
  drawFrame();
  drawText("LOADING FACE TRACKER...", W / 2, H / 2, "orange");
  tracker = await HeadTracker.create();
  ball = spawnOnHumanSide();                                   // NEW
  lastTime = performance.now();                                // NEW
  requestAnimationFrame(loop);                                 // NEW (was loop())
}

start().catch((err) => {
  drawText("Error: " + err.message, W / 2, H / 2, "red");
});