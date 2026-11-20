import {
  Ball, GRAVITY, BALL_R, HEAD_R, BASE_BOUNCE, POWER_GAIN, MAX_POWER_BONUS, STEER,
} from "./physics.js";

const BOT_FY = 0.55;          // bot head height (fraction of screen height)
const BOT_HOME_FX = 0.75;     // where the bot waits when the ball is on your side
const BOT_SPEED = 750.0;      // max head speed, px/s
const BOT_NOD = 600.0;        // fake upward head speed at contact -> header power
const BOT_BONUS = Math.min(BOT_NOD * POWER_GAIN, MAX_POWER_BONUS);
const SIM_DT = 1.0 / 120.0;   // "imagine the future" time step
const SIM_HORIZON = 3.0;      // seconds ahead the bot looks
const MAX_AIM_SHIFT = 260.0;  // at 100% difficulty, aim this many px away from your head

export class RuleBot {
  constructor(W, H, difficulty = 0.0) {
    this.W = W;
    this.H = H;
    this.mid = W / 2;
    this.x = W * BOT_HOME_FX;
    this.y = H * BOT_FY;
    this.lastHit = 0;
    this.name = "RULE";
    this.color = "rgb(70, 130, 255)";
    this.difficulty = difficulty;
    this.aimShift = null;     // picked once per pass, so the bot doesn't jitter
  }

  // 1. look into the future: copy the ball, step it with the SAME physics
  predictLanding(ball, contactY) {
    const sim = new Ball(ball.x, ball.y);
    sim.vx = ball.vx;
    sim.vy = ball.vy;
    const steps = Math.floor(SIM_HORIZON / SIM_DT);
    for (let i = 0; i < steps; i++) {
      sim.step(SIM_DT, this.W);
      if (sim.vy > 0 && sim.y >= contactY) return sim.x;
    }
    return null;
  }

  // 2. decide how off-centre to hit so the pass lands near your head
  aimOffset(fromX, humanHead) {
    let [tx, ty] = humanHead ? humanHead : [this.W * 0.25, this.y];
    // adaptive difficulty: aim AWAY from your head (picked once per pass)
    if (this.aimShift === null) {
      const side = Math.random() < 0.5 ? -1 : 1;
      this.aimShift = side * this.difficulty * MAX_AIM_SHIFT;
    }
    tx = Math.max(HEAD_R, Math.min(this.mid - HEAD_R, tx + this.aimShift));

    const reach = BALL_R + HEAD_R;
    const y0 = this.y - reach * 0.9;          // where the ball leaves the bot's head
    const y1 = ty - reach * 0.9;              // where you will meet it
    const vy0 = -(BASE_BOUNCE + BOT_BONUS);   // launch speed (up is negative)
    // solve y0 + vy0*t + 0.5*g*t^2 = y1, take the later (falling) root
    const disc = vy0 * vy0 - 2 * GRAVITY * (y0 - y1);
    const t = (-vy0 + Math.sqrt(Math.max(disc, 0))) / GRAVITY;
    const vxNeeded = (tx - fromX) / t;
    return Math.max(-1.0, Math.min(-0.15, vxNeeded / STEER));
  }

  // 3. move
  update(dt, ball, humanHead) {
    const reach = BALL_R + HEAD_R;
    let target = this.W * BOT_HOME_FX;

    // ball is on your side -> next pass gets a fresh aim shift
    if (ball.x < this.mid) this.aimShift = null;

    let landX = this.predictLanding(ball, this.y - reach * 0.9);
    if (landX !== null && landX > this.mid) {
      const offset = this.aimOffset(landX, humanHead);
      const contactDy = Math.sqrt(reach ** 2 - (offset * HEAD_R) ** 2);
      const landX2 = this.predictLanding(ball, this.y - contactDy);
      if (landX2 !== null) landX = landX2;
      target = landX - offset * HEAD_R;
    }

    target = Math.max(this.mid + HEAD_R * 0.5, Math.min(this.W - HEAD_R, target));
    const step = BOT_SPEED * dt;
    this.x += Math.max(-step, Math.min(step, target - this.x));
    return [this.x, this.y];
  }

  velocity() {
    return [0.0, -BOT_NOD];   // the bot "nods" upward so its header has power
  }

  draw(ctx, ball) {
    const cx = this.x, cy = this.y;
    // head
    ctx.beginPath();
    ctx.arc(cx, cy, HEAD_R, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = "white";
    ctx.stroke();
    // eyes follow the ball
    const ang = Math.atan2(ball.y - cy, ball.x - cx);
    for (const ex of [cx - 22, cx + 22]) {
      const ey = cy - 12;
      ctx.beginPath();
      ctx.arc(ex, ey, 12, 0, Math.PI * 2);
      ctx.fillStyle = "white";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(ex + 5 * Math.cos(ang), ey + 5 * Math.sin(ang), 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgb(10, 10, 10)";
      ctx.fill();
    }
    // smile
    ctx.beginPath();
    ctx.ellipse(cx, cy + 18, 22, 13, 0, 0.1 * Math.PI, 0.9 * Math.PI);
    ctx.strokeStyle = "white";
    ctx.stroke();
  }
}