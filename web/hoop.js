import { Ball, BALL_R } from "./physics.js";

export const GAME_TIME = 60.0; // seconds per round
const HOOP_FX = 0.72,
    HOOP_FY = 0.24; // rim centre (fractions of screen)
const RIM_W = 130; // rim opening width
const BOARD_H = 95; // backboard height
const RIM_END_R = 7; // rim-end collider radius (causes rattle-outs)
const MOVING_HOOP = false; // true = sliding hoop (hard mode)
const HOOP_SPEED = 90.0; // px/s when moving

function uniform(a, b) {
    return a + Math.random() * (b - a);
}

export class Hoop {
    constructor(W, H) {
        this.W = W;
        this.x = W * HOOP_FX;
        this.y = H * HOOP_FY;
        this.dir = 1;
    }

    update(dt) {
        if (!MOVING_HOOP) return;
        this.x += this.dir * HOOP_SPEED * dt;
        const half = RIM_W / 2;
        if (this.x - half < this.W * 0.45) {
            this.x = this.W * 0.45 + half;
            this.dir = 1;
        } else if (this.x + half > this.W - 10) {
            this.x = this.W - 10 - half;
            this.dir = -1;
        }
    }

    // like @property in Python: used as hoop.left, no ()
    get left() { return this.x - RIM_W / 2; }
    get right() { return this.x + RIM_W / 2; }

    // bounce the ball off the two rim ends and the backboard
    collide(ball) {
        for (const ex of[this.left, this.right]) {
            const dx = ball.x - ex,
                dy = ball.y - this.y;
            const d = Math.hypot(dx, dy);
            if (d < BALL_R + RIM_END_R && d > 1e-3) {
                const nx = dx / d,
                    ny = dy / d;
                ball.x = ex + nx * (BALL_R + RIM_END_R);
                ball.y = this.y + ny * (BALL_R + RIM_END_R);
                const dot = ball.vx * nx + ball.vy * ny; // reflect the velocity
                ball.vx -= 2 * dot * nx;
                ball.vy -= 2 * dot * ny;
                ball.vx *= 0.7;
                ball.vy *= 0.7;
            }
        }
        // backboard: vertical wall just behind the right rim end
        const boardX = this.right + 4;
        const boardTop = this.y - BOARD_H,
            boardBot = this.y + 8;
        if (boardTop <= ball.y && ball.y <= boardBot &&
            ball.prevX + BALL_R <= boardX && boardX < ball.x + BALL_R &&
            ball.vx > 0) {
            ball.x = boardX - BALL_R;
            ball.vx = -Math.abs(ball.vx) * 0.7;
        }
    }

    // true on a clean downward pass through the rim (after a header)
    scored(ball) {
        const crossed = ball.prevY < this.y && this.y <= ball.y;
        const inside = this.left + BALL_R * 0.4 < ball.x && ball.x < this.right - BALL_R * 0.4;
        return crossed && inside && ball.vy > 0 && ball.headed;
    }

    draw(ctx) {
        // backboard
        ctx.fillStyle = "rgb(230, 230, 230)";
        ctx.beginPath();
        ctx.roundRect(this.right + 4, this.y - BOARD_H, 8, BOARD_H, 3);
        ctx.fill();
        // rim
        ctx.strokeStyle = "rgb(255, 140, 30)";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(this.left, this.y);
        ctx.lineTo(this.right, this.y);
        ctx.stroke();
        ctx.fillStyle = "rgb(255, 140, 30)";
        for (const ex of[this.left, this.right]) {
            ctx.beginPath();
            ctx.arc(ex, this.y, RIM_END_R, 0, Math.PI * 2);
            ctx.fill();
        }
        // little net hint
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1;
        for (let k = 1; k < 5; k++) {
            const fx = this.left + ((this.right - this.left) * k) / 5;
            ctx.beginPath();
            ctx.moveTo(fx, this.y);
            ctx.lineTo(this.x, this.y + 34);
            ctx.stroke();
        }
    }
}

// drop the ball on the opposite side from the hoop, so it must be headed in
export function spawnHoopsBall(W, H, hoop) {
    const fx = hoop.x > W / 2 ? uniform(0.15, 0.35) : uniform(0.65, 0.85);
    const b = new Ball(W * fx, H * 0.15);
    b.vx = uniform(-40, 40);
    return b;
}