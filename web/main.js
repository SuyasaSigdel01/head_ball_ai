import { HeadTracker } from "./tracker.js";
import { Ball, BALL_R, HEAD_R, MAX_DT, HIT_COOLDOWN, tryHeader } from "./physics.js";
import { RuleBot } from "./bot.js";

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const W = canvas.width, H = canvas.height;
const mid = W / 2;
const RALLY_TARGET = 5;                       // NEW: rally 5, 10, 15... -> harder

const GREEN = "rgb(60, 220, 120)";            // NEW: same colours as Python
const RED = "rgb(240, 80, 80)";
const YELLOW = "rgb(250, 210, 60)";
const ORANGE = "rgb(255, 140, 30)";

let tracker = null;
let ball = null;
let bot = null;
let lastTime = 0;
let humanLastHit = 0;

let rally = 0, best = 0, prevRally = 0;       // NEW
let lastTouch = null;                         // NEW: "human" / "bot", who headed it last
let difficulty = 0.3;                         // NEW: start easy
let msg = "", msgUntil = 0;                   // NEW
let state = "WAIT";                           // WAIT -> PLAY, like Python

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

function drawText(text, x, y, color, size = 30, align = "center") {   // CHANGED: align
  ctx.font = `bold ${size}px Consolas, monospace`;
  ctx.textAlign = align;
  ctx.fillStyle = "black";
  ctx.fillText(text, x + 2, y + 2);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function drawMidline() {
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  ctx.setLineDash([14, 14]);
  ctx.beginPath();
  ctx.moveTo(mid, 0);
  ctx.lineTo(mid, H);
  ctx.stroke();
  ctx.setLineDash([]);
}

function setMsg(text, seconds) {              // NEW: like msg, msg_until in Python
  msg = text;
  msgUntil = performance.now() / 1000 + seconds;
}

function startRally() {
  rally = 0;
  lastTouch = null;
  ball = spawnOnHumanSide();
  state = "PLAY";
}

// SPACE / R start or restart, like Python
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" || e.key === "r" || e.key === "R") {
    e.preventDefault();       // stop SPACE from scrolling the page
    startRally();
  }
});

// a click also starts (handy on a laptop touchpad), but only from the start screen
canvas.addEventListener("click", () => {
  if (state === "WAIT") startRally();
});

function drawHud() {                          // NEW
  drawText(`RALLY ${rally}`, 16, 40, ORANGE, 30, "left");
  drawText(`BEST ${best}`, 16, 76, YELLOW, 30, "left");

  const pct = Math.round(difficulty * 100);
  let label = "HARD", col = RED;
  if (difficulty < 0.45) { label = "EASY"; col = GREEN; }
  else if (difficulty < 0.75) { label = "MEDIUM"; col = YELLOW; }
  drawText(`DIFF ${pct}% (${label})`, 16, 108, col, 22, "left");

  drawText("YOU", W / 4, 40, "white");
  drawText(`${bot.name} BOT`, (3 * W) / 4, 40, bot.color);

  if (performance.now() / 1000 < msgUntil) drawText(msg, W / 2, H / 2 + 80, RED);
}

function loop(time) {
  const dt = Math.min((time - lastTime) / 1000, MAX_DT);
  lastTime = time;

  drawFrame();
  drawMidline();
  const head = tracker.update(video, W, H);

  if (state === "PLAY") {
    if (!head) {
      drawText("FACE NOT DETECTED", W / 2, 70, RED);
    } else {
      ball.step(dt, W);
      const now = performance.now() / 1000;

      // you: only count while on the LEFT half
      if (head[0] < mid && now - humanLastHit > HIT_COOLDOWN
          && tryHeader(ball, head, tracker.velocity())) {
        humanLastHit = now;
        if (lastTouch === "bot") rally += 1;
        lastTouch = "human";
      }

      // bot: same tryHeader(), same physics as you
      const botPos = bot.update(dt, ball, head);
      if (now - bot.lastHit > HIT_COOLDOWN && tryHeader(ball, botPos, bot.velocity())) {
        bot.lastHit = now;
        if (lastTouch === "human") rally += 1;
        lastTouch = "bot";
      }

      best = Math.max(best, rally);

      // rally just reached 5, 10, 15... -> harder right away
      if (rally !== prevRally && rally > 0 && rally % RALLY_TARGET === 0) {
        difficulty = Math.min(1.0, difficulty + 0.1);
        bot.difficulty = difficulty;
        setMsg(`LEVEL UP!  DIFF ${Math.round(difficulty * 100)}%`, 1.5);
      }
      prevRally = rally;

      // ball hit the floor -> rally over
      if (ball.y - BALL_R > H) {
        const who = ball.x < mid ? "YOU" : "BOT";
        setMsg(`${who} DROPPED IT  -  rally ${rally}`, 1.5);
        if (who === "YOU") {
          difficulty = Math.max(0.0, difficulty - 0.1);
          bot.difficulty = difficulty;
        }
        rally = 0;
        lastTouch = null;
        ball = spawnOnHumanSide();
      }
    }
  }

  // ---------------- draw ----------------
  if (head) {
    ctx.strokeStyle = head[0] < mid ? GREEN : RED;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2);
    ctx.stroke();
  }
  bot.draw(ctx, ball);
  if (state === "PLAY") ball.draw(ctx);
  drawHud();

  if (state === "WAIT") {
    drawText("HEADER PASS", W / 2, H / 2 - 40, ORANGE, 60);
    drawText("stand on the LEFT  -  SPACE or click to start", W / 2, H / 2 + 20, "white");
  }

  requestAnimationFrame(loop);
}
async function start() {
  await startCamera();
  drawFrame();
  drawText("LOADING FACE TRACKER...", W / 2, H / 2, "orange");
  tracker = await HeadTracker.create();
  ball = spawnOnHumanSide();
  bot = new RuleBot(W, H, difficulty);                         // CHANGED: start at 30%
  lastTime = performance.now();
  requestAnimationFrame(loop);
}

start().catch((err) => {
  drawText("Error: " + err.message, W / 2, H / 2, "red");
});