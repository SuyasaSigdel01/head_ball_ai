const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "orange";
ctx.font = "bold 60px Consolas, monospace";
ctx.textAlign = "center";
ctx.fillText("HEADER PASS", canvas.width / 2, canvas.height / 2);