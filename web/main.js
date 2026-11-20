const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const W = canvas.width, H = canvas.height;

// 1. ask the browser for the webcam (like cv2.VideoCapture)
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

// 2. draw the frame mirrored (like cv2.flip(frame, 1))
function drawFrame() {
  ctx.save();
  ctx.translate(W, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, W, H);
  ctx.restore();
}

// 3. game loop: the browser calls this ~60 times a second (like while running:)
function loop() {
  drawFrame();
  requestAnimationFrame(loop);
}

startCamera()
  .then(loop)
  .catch((err) => {
    ctx.fillStyle = "red";
    ctx.font = "bold 30px Consolas, monospace";
    ctx.textAlign = "center";
    ctx.fillText("Camera error: " + err.message, W / 2, H / 2);
  });