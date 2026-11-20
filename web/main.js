import { HeadTracker } from "./tracker.js";
import { Ball, BALL_R, HEAD_R, MAX_DT, HIT_COOLDOWN, tryHeader } from "./physics.js";
import { RuleBot } from "./bot.js";                                    // NEW

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const W = canvas.width, H = canvas.height;
const mid = W / 2;                                                     // NEW

let tracker = null;
let ball = null;
let bot = null;                                                        // NEW
let lastTime = 0;
let humanLastHit = 0;

function uniform(a, b) {
  return a + Math.random() * (b - a);
}

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

function drawMidline() {                                               // NEW
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  ctx.setLineDash([14, 14]);
  ctx.beginPath();
  ctx.moveTo(mid, 0);
  ctx.lineTo(mid, H);
  ctx.stroke();
  ctx.setLineDash([]);
}

function loop(time) {
  const dt = Math.min((time - lastTime) / 1000, MAX_DT);
  lastTime = time;

  drawFrame();
  drawMidline();                                                       // NEW
  const head = tracker.update(video, W, H);

  if (head) {
    ball.step(dt, W);
    const now = performance.now() / 1000;

    // you: only count while on the LEFT half                          // CHANGED
    if (head[0] < mid && now - humanLastHit > HIT_COOLDOWN
        && tryHeader(ball, head, tracker.velocity())) {
      humanLastHit = now;
    }

    // bot: same tryHeader(), same physics as you                      // NEW
    const botPos = bot.update(dt, ball, head);
    if (now - bot.lastHit > HIT_COOLDOWN && tryHeader(ball, botPos, bot.velocity())) {
      bot.lastHit = now;
    }

    if (ball.y - BALL_R > H) ball = spawnOnHumanSide();

    // green on your side, red if you cross the line                   // CHANGED
    ctx.strokeStyle = head[0] < mid ? "rgb(60, 220, 120)" : "rgb(240, 80, 80)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    drawText("FACE NOT DETECTED", W / 2, 70, "rgb(240, 80, 80)");
  }

  bot.draw(ctx, ball);                                                 // NEW
  ball.draw(ctx);
  requestAnimationFrame(loop);
}

async function start() {
  await startCamera();
  drawFrame();
  drawText("LOADING FACE TRACKER...", W / 2, H / 2, "orange");
  tracker = await HeadTracker.create();
  ball = spawnOnHumanSide();
  bot = new RuleBot(W, H);                                             // NEW
  lastTime = performance.now();
  requestAnimationFrame(loop);
}

start().catch((err) => {
  drawText("Error: " + err.message, W / 2, H / 2, "red");
});