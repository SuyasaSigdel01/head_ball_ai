import { FaceLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const FOREHEAD_LM = 10;   // same landmark as Python
const SMOOTH_N = 5;       // average the last 5 positions (less jitter)
const VEL_WINDOW = 4;     // head speed from the last 4 samples

export class HeadTracker {
  // loading the model takes time, so we build the tracker with an async helper
  static async create() {
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const landmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
    });
    return new HeadTracker(landmarker);
  }

  constructor(landmarker) {
    this.landmarker = landmarker;
    this.history = [];        // like deque(maxlen=SMOOTH_N)
    this.samples = [];        // like deque(maxlen=VEL_WINDOW)
    this.lastVideoTime = -1;
    this.head = null;
  }

  // returns [x, y] in canvas pixels, or null if no face
  update(video, W, H) {
    // the game runs at ~60 fps but the camera at ~30: skip repeated frames
    if (video.currentTime === this.lastVideoTime) return this.head;
    this.lastVideoTime = video.currentTime;

    const now = performance.now();
    const res = this.landmarker.detectForVideo(video, now);
    if (res.faceLandmarks.length === 0) {
      this.history = [];
      this.samples = [];
      this.head = null;
      return null;
    }

    const lm = res.faceLandmarks[0][FOREHEAD_LM];
    this.history.push([(1 - lm.x) * W, lm.y * H]);   // 1 - x = mirror, like cv2.flip
    if (this.history.length > SMOOTH_N) this.history.shift();

    const sx = this.history.reduce((s, p) => s + p[0], 0) / this.history.length;
    const sy = this.history.reduce((s, p) => s + p[1], 0) / this.history.length;

    this.samples.push([now / 1000, sx, sy]);
    if (this.samples.length > VEL_WINDOW) this.samples.shift();

    this.head = [sx, sy];
    return this.head;
  }

  velocity() {
    if (this.samples.length < 2) return [0, 0];
    const [t0, x0, y0] = this.samples[0];
    const [t1, x1, y1] = this.samples[this.samples.length - 1];
    const dt = t1 - t0;
    return dt <= 1e-4 ? [0, 0] : [(x1 - x0) / dt, (y1 - y0) / dt];
  }
}