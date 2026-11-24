import { HeadTracker, coverCrop } from "./tracker.js";
import { Ball, BALL_R, HEAD_R, MAX_DT, HIT_COOLDOWN, tryHeader } from "./physics.js";
import { RuleBot } from "./bot.js";
import { Hoop, spawnHoopsBall, GAME_TIME } from "./hoop.js";

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const W = canvas.width,
    H = canvas.height;
const mid = W / 2;
const RALLY_TARGET = 5;

const GREEN = "rgb(60, 220, 120)";
const RED = "rgb(240, 80, 80)";
const YELLOW = "rgb(250, 210, 60)";
const ORANGE = "rgb(255, 140, 30)";

let tracker = null;
let ball = null;
let bot = null;
let hoop = null;
let lastTime = 0;
let humanLastHit = 0;

let mode = "PASS"; // "PASS" or "HOOPS"
let state = "WAIT"; // WAIT -> PLAY -> OVER (OVER is hoops only)

// pass mode
let rally = 0,
    best = 0,
    prevRally = 0;
let lastTouch = null;
let difficulty = 0.3;
let msg = "",
    msgUntil = 0;

// hoops mode
let score = 0,
    hoopsBest = 0;
let timeLeft = GAME_TIME;
let flashUntil = 0;

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
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
    });
    video.srcObject = stream;
    await video.play();
}

// mirrored camera frame, cropped (not stretched) to fill the 16:9 canvas
function drawFrame() {
    const c = coverCrop(video, W, H);
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, c.sx, c.sy, c.sw, c.sh, 0, 0, W, H);
    ctx.restore();
}

function drawText(text, x, y, color, size = 30, align = "center") {
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

function drawHead(head, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(head[0], head[1], HEAD_R, 0, Math.PI * 2);
    ctx.stroke();
}

function setMsg(text, seconds) {
    msg = text;
    msgUntil = performance.now() / 1000 + seconds;
}

// keep the screen on while playing (for the website; the Android app does it natively)
async function keepScreenOn() {
    try {
        if ("wakeLock" in navigator) await navigator.wakeLock.request("screen");
    } catch (err) {
        // not supported or not allowed: the game still works, the screen may just dim
    }
}

// ---------------- starting a mode ----------------
function startPass() {
    keepScreenOn();
    mode = "PASS";
    rally = 0;
    lastTouch = null;
    ball = spawnOnHumanSide();
    state = "PLAY";
}

function startHoops() {
    keepScreenOn();
    mode = "HOOPS";
    score = 0;
    timeLeft = GAME_TIME;
    ball = spawnHoopsBall(W, H, hoop);
    state = "PLAY";
}

// keyboard
window.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();
    const go = e.code === "Space" || key === "r";
    if (e.code === "Space") e.preventDefault();

    if (key === "escape") {
        state = "WAIT"; // back to the menu
    } else if (state === "WAIT") {
        if (e.code === "Space") startPass();
        else if (key === "h") startHoops();
    } else if (mode === "PASS" && go) {
        startPass(); // restart the rally
    } else if (mode === "HOOPS" && state === "OVER" && go) {
        startHoops(); // play again
    }
});

// tap / click: phones have no keyboard
canvas.addEventListener("click", (e) => {
    const x = (e.offsetX / canvas.clientWidth) * W; // screen px -> canvas px
    if (state === "WAIT") {
        if (x < mid) startPass();
        else startHoops();
    } else if (state === "OVER") {
        startHoops();
    }
});

// ---------------- game logic ----------------
function updatePass(dt, head) {
    ball.step(dt, W);
    const now = performance.now() / 1000;

    // you: only count while on the LEFT half
    if (head[0] < mid && now - humanLastHit > HIT_COOLDOWN &&
        tryHeader(ball, head, tracker.velocity())) {
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

function updateHoops(dt, head) {
    timeLeft -= dt;
    hoop.update(dt);
    ball.step(dt, W);
    hoop.collide(ball);

    const now = performance.now() / 1000;
    if (now - humanLastHit > HIT_COOLDOWN && tryHeader(ball, head, tracker.velocity())) {
        humanLastHit = now;
    }

    if (hoop.scored(ball)) {
        score += 1;
        hoopsBest = Math.max(hoopsBest, score);
        flashUntil = now + 0.4;
        ball = spawnHoopsBall(W, H, hoop);
    } else if (ball.y - BALL_R > H) {
        ball = spawnHoopsBall(W, H, hoop); // missed, hit the floor
    }

    if (timeLeft <= 0) {
        timeLeft = 0;
        state = "OVER";
    }
}

// ---------------- drawing ----------------
function drawPassHud() {
    drawText(`RALLY ${rally}`, 16, 40, ORANGE, 30, "left");
    drawText(`BEST ${best}`, 16, 76, YELLOW, 30, "left");

    const pct = Math.round(difficulty * 100);
    let label = "HARD",
        col = RED;
    if (difficulty < 0.45) { label = "EASY";
        col = GREEN; } else if (difficulty < 0.75) { label = "MEDIUM";
        col = YELLOW; }
    drawText(`DIFF ${pct}% (${label})`, 16, 108, col, 22, "left");

    drawText("YOU", W / 4, 40, "white");
    drawText(`${bot.name} BOT`, (3 * W) / 4, 40, bot.color);

    if (performance.now() / 1000 < msgUntil) drawText(msg, W / 2, H / 2 + 80, RED);
}

function drawHoopsHud() {
    drawText(`BASKETS ${score}`, 16, 40, ORANGE, 30, "left");
    drawText(`BEST ${hoopsBest}`, 16, 76, YELLOW, 30, "left");
    if (state === "PLAY") {
        const col = timeLeft < 10 ? RED : "white";
        drawText(`${timeLeft.toFixed(1).padStart(4, "0")}s`, W - 16, 40, col, 30, "right");
    }
    if (performance.now() / 1000 < flashUntil) drawText("SCORE!", W / 2, H / 2, GREEN, 60);
}

function drawMenu() {
    drawText("HEAD BALL AI", W / 2, H / 2 - 60, ORANGE, 60);
    drawText("SPACE or tap LEFT  -  pass with the bot", W / 2, H / 2 + 10, "white");
    drawText("H or tap RIGHT  -  Header Hoops", W / 2, H / 2 + 55, "white");
}

function loop(time) {
    const dt = Math.min((time - lastTime) / 1000, MAX_DT);
    lastTime = time;

    drawFrame();
    const head = tracker.update(video, W, H);

    // update (no face -> everything pauses, like Python)
    if (state === "PLAY" && head) {
        if (mode === "PASS") updatePass(dt, head);
        else updateHoops(dt, head);
    }

    // draw
    if (state === "WAIT") {
        if (head) drawHead(head, GREEN);
        drawMenu();
    } else if (mode === "PASS") {
        drawMidline();
        if (head) drawHead(head, head[0] < mid ? GREEN : RED);
        bot.draw(ctx, ball);
        ball.draw(ctx);
        drawPassHud();
    } else {
        hoop.draw(ctx);
        if (head) drawHead(head, GREEN);
        if (state === "PLAY") ball.draw(ctx);
        drawHoopsHud();
        if (state === "OVER") {
            drawText(`${score} BASKETS`, W / 2, H / 2 - 40, ORANGE, 60);
            drawText("SPACE or tap to play again  -  ESC: menu", W / 2, H / 2 + 20, "white");
        }
    }

    if (state === "PLAY" && !head) drawText("FACE NOT DETECTED", W / 2, 120, RED);

    requestAnimationFrame(loop);
}

async function start() {
    await startCamera();
    drawFrame();
    drawText("LOADING FACE TRACKER...", W / 2, H / 2, "orange");
    tracker = await HeadTracker.create();
    bot = new RuleBot(W, H, difficulty);
    hoop = new Hoop(W, H);
    ball = spawnOnHumanSide();
    lastTime = performance.now();
    requestAnimationFrame(loop);
}

start().catch((err) => {
    drawText("Error: " + err.message, W / 2, H / 2, "red");
});