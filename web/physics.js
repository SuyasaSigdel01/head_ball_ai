// ---- tunables: same numbers as header_game.py ----
export const GRAVITY = 1400.0;
export const BALL_R = 24;
export const HEAD_R = 62;
export const BASE_BOUNCE = 680.0;      // header power
export const POWER_GAIN = 0.5;
export const MAX_POWER_BONUS = 560.0;
export const STEER = 520.0;            // sideways aim from off-centre contact
export const HIT_COOLDOWN = 0.18;
export const MAX_DT = 1.0 / 30.0;

export class Ball {
  constructor(x, y) {
    this.reset(x, y);
  }

  reset(x, y) {
    this.x = x;
    this.y = y;
    this.prevX = x;
    this.prevY = y;
    this.vx = 0;
    this.vy = 0;
    this.headed = false;
  }

  step(dt, w) {
    this.prevX = this.x;
    this.prevY = this.y;
    this.vy += GRAVITY * dt;
    this.x += this.vx * dt;
    this.y += this.vy * dt;
    // bounce off the side walls
    if (this.x < BALL_R) {
      this.x = BALL_R;
      this.vx = Math.abs(this.vx) * 0.85;
    } else if (this.x > w - BALL_R) {
      this.x = w - BALL_R;
      this.vx = -Math.abs(this.vx) * 0.85;
    }
    // bounce off the ceiling
    if (this.y < BALL_R) {
      this.y = BALL_R;
      this.vy = Math.abs(this.vy) * 0.5;
    }
  }

  draw(ctx) {
    ctx.beginPath();
    ctx.arc(this.x, this.y, BALL_R, 0, Math.PI * 2);
    ctx.fillStyle = "rgb(255, 140, 30)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgb(10, 10, 10)";
    ctx.stroke();
    // basketball seams
    ctx.beginPath();
    ctx.moveTo(this.x - BALL_R, this.y);
    ctx.lineTo(this.x + BALL_R, this.y);
    ctx.moveTo(this.x, this.y - BALL_R);
    ctx.lineTo(this.x, this.y + BALL_R);
    ctx.stroke();
  }
}

// Bounce the ball off a head if they touch. Returns true on a hit.
// Shared by the human and the bot, so the physics are identical for both.
// Cooldown is NOT checked here: each player keeps its own lastHit time.
export function tryHeader(ball, headPos, headVel) {
  // 1. distance between ball centre and head centre
  const dx = ball.x - headPos[0];
  const dy = ball.y - headPos[1];
  const dist = Math.hypot(dx, dy);

  // 2. no contact, or ball going up (already headed) -> no hit
  if (dist >= BALL_R + HEAD_R || ball.vy <= 0) return false;

  // 3. sideways aim from off-centre contact, power from upward head speed
  const offset = Math.max(-1, Math.min(1, dx / HEAD_R));
  ball.vx = offset * STEER;
  const bonus = Math.min(Math.max(-headVel[1], 0) * POWER_GAIN, MAX_POWER_BONUS);
  ball.vy = -(BASE_BOUNCE + bonus);

  // 4. push the ball just outside the head so they don't overlap next frame
  const d = Math.max(dist, 1e-3);
  ball.x = headPos[0] + (dx / d) * (BALL_R + HEAD_R + 1);
  ball.y = headPos[1] + (dy / d) * (BALL_R + HEAD_R + 1);

  ball.headed = true;
  return true;
}