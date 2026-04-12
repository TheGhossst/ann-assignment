const CANVAS_SIZE = 360;
const BRUSH_SIZE = 22;
const MODEL_URL = "lenet5_weights.json";
const MNIST_MEAN = 0.1307;
const MNIST_STD = 0.3081;

const canvas = document.getElementById("digit-canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const modelStatus = document.getElementById("model-status");
const modelPanel = document.querySelector(".panel.canvas-panel");
const clearButton = document.getElementById("clear-button");
const predictionDigit = document.getElementById("prediction-digit");
const predictionConfidence = document.getElementById("prediction-confidence");
const predictionChip = document.getElementById("prediction-chip");
const probabilityContainer = document.getElementById("probabilities");

const cropCanvas = document.createElement("canvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });
const resizeCanvas = document.createElement("canvas");
const resizeCtx = resizeCanvas.getContext("2d", { willReadFrequently: true });

const probabilityRows = [];

let model = null;
let modelReady = false;
let isDrawing = false;
let lastPoint = null;
let predictionTimer = null;

canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;
ctx.lineCap = "round";
ctx.lineJoin = "round";
ctx.lineWidth = BRUSH_SIZE;

createProbabilityRows();
clearCanvas();
bindEvents();
void loadModel();

function bindEvents() {
  canvas.addEventListener("pointerdown", handlePointerDown);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerup", handlePointerEnd);
  canvas.addEventListener("pointercancel", handlePointerEnd);
  canvas.addEventListener("pointerleave", handlePointerEnd);
  clearButton.addEventListener("click", () => {
    clearCanvas();
    schedulePrediction(0);
  });

  window.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (
      event.key === "Escape" ||
      event.key === "Backspace" ||
      event.key.toLowerCase() === "c"
    ) {
      event.preventDefault();
      clearCanvas();
      schedulePrediction(0);
    }
  });
}

function createProbabilityRows() {
  probabilityContainer.innerHTML = "";
  for (let digit = 0; digit < 10; digit += 1) {
    const row = document.createElement("div");
    row.className = "probability-row";
    row.dataset.digit = String(digit);
    row.innerHTML = `
      <span class="probability-digit">${digit}</span>
      <div class="probability-track">
        <div class="probability-fill"></div>
      </div>
      <span class="probability-value">0.0%</span>
    `;
    probabilityContainer.appendChild(row);
    probabilityRows.push({
      row,
      fill: row.querySelector(".probability-fill"),
      value: row.querySelector(".probability-value"),
    });
  }
}

function setModelState(message, state = "loading") {
  modelStatus.textContent = message;
  modelPanel.classList.remove("is-ready", "is-error");
  if (state === "ready") modelPanel.classList.add("is-ready");
  else if (state === "error") modelPanel.classList.add("is-error");
}

function clearCanvas() {
  ctx.fillStyle = "#0d1520";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.strokeStyle = "#e8f4ff";
  ctx.fillStyle = "#e8f4ff";
  ctx.lineWidth = BRUSH_SIZE;
  renderIdlePrediction();
}

function renderIdlePrediction() {
  predictionDigit.textContent = "—";
  predictionDigit.classList.remove("has-value");
  predictionConfidence.textContent = "Waiting for a digit.";
  predictionChip.textContent = modelReady ? "Draw to update" : "Loading model";
  updateProbabilityDisplay(new Float32Array(10), -1);
}

function handlePointerDown(event) {
  if (event.button !== 0) return;
  canvas.setPointerCapture(event.pointerId);
  isDrawing = true;
  lastPoint = pointerToCanvasPoint(event);
  stamp(lastPoint);
  schedulePrediction(160);
}

function handlePointerMove(event) {
  if (!isDrawing) return;
  const point = pointerToCanvasPoint(event);
  drawSegment(lastPoint, point);
  lastPoint = point;
  schedulePrediction(140);
}

function handlePointerEnd(event) {
  if (!isDrawing) return;
  isDrawing = false;
  lastPoint = null;
  schedulePrediction(40);
  if (canvas.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
}

function pointerToCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * CANVAS_SIZE,
    y: ((event.clientY - rect.top) / rect.height) * CANVAS_SIZE,
  };
}

function stamp(point) {
  ctx.beginPath();
  ctx.arc(point.x, point.y, BRUSH_SIZE / 2, 0, Math.PI * 2);
  ctx.fill();
}

function drawSegment(startPoint, endPoint) {
  if (!startPoint || !endPoint) return;
  ctx.beginPath();
  ctx.moveTo(startPoint.x, startPoint.y);
  ctx.lineTo(endPoint.x, endPoint.y);
  ctx.stroke();
}

function schedulePrediction(delay) {
  if (!modelReady) return;
  window.clearTimeout(predictionTimer);
  predictionTimer = window.setTimeout(() => {
    void predictCanvas();
  }, delay);
}

async function loadModel() {
  try {
    setModelState("Loading checkpoint…", "loading");
    const response = await fetch(MODEL_URL, { cache: "no-store" });
    if (!response.ok)
      throw new Error(`Failed to load ${MODEL_URL}: ${response.status}`);
    const snapshot = await response.json();
    model = unpackModel(snapshot.parameters);
    modelReady = true;
    setModelState("LeNet-5 ready", "ready");
    schedulePrediction(0);
  } catch (error) {
    console.error(error);
    modelReady = false;
    model = null;
    setModelState("Model failed to load", "error");
    predictionDigit.textContent = "!";
    predictionConfidence.textContent =
      "Serve this page via a local server so the checkpoint can load.";
    predictionChip.textContent = "Load failed";
  }
}

function unpackModel(parameters) {
  return {
    conv1: {
      weight: toFloat32(parameters["feature_extractor.0.weight"]),
      bias: toFloat32(parameters["feature_extractor.0.bias"]),
      inChannels: 1,
      outChannels: 6,
      kernelSize: 5,
      padding: 2,
    },
    conv2: {
      weight: toFloat32(parameters["feature_extractor.3.weight"]),
      bias: toFloat32(parameters["feature_extractor.3.bias"]),
      inChannels: 6,
      outChannels: 16,
      kernelSize: 5,
      padding: 0,
    },
    conv3: {
      weight: toFloat32(parameters["feature_extractor.6.weight"]),
      bias: toFloat32(parameters["feature_extractor.6.bias"]),
      inChannels: 16,
      outChannels: 120,
      kernelSize: 5,
      padding: 0,
    },
    fc1: {
      weight: toFloat32(parameters["classifier.0.weight"]),
      bias: toFloat32(parameters["classifier.0.bias"]),
      inFeatures: 120,
      outFeatures: 84,
    },
    fc2: {
      weight: toFloat32(parameters["classifier.2.weight"]),
      bias: toFloat32(parameters["classifier.2.bias"]),
      inFeatures: 84,
      outFeatures: 10,
    },
  };
}

function toFloat32(values) {
  return Float32Array.from(values.flat(Infinity));
}

async function predictCanvas() {
  if (!modelReady || !model) return;
  const tensor = preprocessCanvas();
  if (!tensor) {
    renderIdlePrediction();
    return;
  }
  const logits = forwardPass(tensor, model);
  const probabilities = softmax(logits);
  const bestDigit = argMax(probabilities);
  renderPrediction(probabilities, bestDigit);
}

function preprocessCanvas() {
  const image = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  const grayscale = new Uint8ClampedArray(CANVAS_SIZE * CANVAS_SIZE);
  let total = 0;

  for (let i = 0, p = 0; i < image.data.length; i += 4, p += 1) {
    const value = image.data[i];
    grayscale[p] = value;
    total += value;
  }

  const invert = total / grayscale.length > 127;
  const processed = new Uint8ClampedArray(grayscale.length);
  let minX = CANVAS_SIZE,
    minY = CANVAS_SIZE,
    maxX = -1,
    maxY = -1;

  for (let y = 0; y < CANVAS_SIZE; y += 1) {
    for (let x = 0; x < CANVAS_SIZE; x += 1) {
      const index = y * CANVAS_SIZE + x;
      const value = invert ? 255 - grayscale[index] : grayscale[index];
      processed[index] = value;
      if (value > 30) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;

  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const padding = Math.max(
    2,
    Math.round(Math.max(cropWidth, cropHeight) * 0.2),
  );
  const paddedWidth = cropWidth + padding * 2;
  const paddedHeight = cropHeight + padding * 2;

  cropCanvas.width = paddedWidth;
  cropCanvas.height = paddedHeight;
  cropCtx.fillStyle = "#000000";
  cropCtx.fillRect(0, 0, paddedWidth, paddedHeight);

  const cropImage = cropCtx.createImageData(paddedWidth, paddedHeight);
  const cropPixels = cropImage.data;

  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const sourceIndex = (minY + y) * CANVAS_SIZE + (minX + x);
      const targetIndex = ((y + padding) * paddedWidth + (x + padding)) * 4;
      const value = processed[sourceIndex];
      cropPixels[targetIndex] = value;
      cropPixels[targetIndex + 1] = value;
      cropPixels[targetIndex + 2] = value;
      cropPixels[targetIndex + 3] = 255;
    }
  }

  cropCtx.putImageData(cropImage, 0, 0);
  resizeCanvas.width = 28;
  resizeCanvas.height = 28;
  resizeCtx.imageSmoothingEnabled = true;
  resizeCtx.clearRect(0, 0, 28, 28);
  resizeCtx.drawImage(
    cropCanvas,
    0,
    0,
    paddedWidth,
    paddedHeight,
    0,
    0,
    28,
    28,
  );

  const resized = resizeCtx.getImageData(0, 0, 28, 28).data;
  const tensor = new Float32Array(28 * 28);
  for (let i = 0, p = 0; i < resized.length; i += 4, p += 1) {
    tensor[p] = (resized[i] / 255 - MNIST_MEAN) / MNIST_STD;
  }
  return tensor;
}

function forwardPass(input, weights) {
  const conv1 = conv2d(input, 1, 28, 28, weights.conv1);
  const act1 = tanhArray(conv1);
  const pool1 = avgPool2x2(act1, 6, 28, 28);
  const conv2 = conv2d(pool1, 6, 14, 14, weights.conv2);
  const act2 = tanhArray(conv2);
  const pool2 = avgPool2x2(act2, 16, 10, 10);
  const conv3 = conv2d(pool2, 16, 5, 5, weights.conv3);
  const act3 = tanhArray(conv3);
  const fc1 = dense(act3, weights.fc1);
  const act4 = tanhArray(fc1);
  return dense(act4, weights.fc2);
}

function conv2d(input, inChannels, height, width, layer) {
  const kernelSize = layer.kernelSize;
  const padding = layer.padding;
  const outHeight = height - kernelSize + 1 + padding * 2;
  const outWidth = width - kernelSize + 1 + padding * 2;
  const output = new Float32Array(layer.outChannels * outHeight * outWidth);
  const kernelArea = kernelSize * kernelSize;
  const inputArea = height * width;
  const outPlaneSize = outHeight * outWidth;

  for (let oc = 0; oc < layer.outChannels; oc += 1) {
    const outputBase = oc * outPlaneSize;
    const biasValue = layer.bias[oc];
    for (let oy = 0; oy < outHeight; oy += 1) {
      for (let ox = 0; ox < outWidth; ox += 1) {
        let sum = biasValue;
        for (let ic = 0; ic < inChannels; ic += 1) {
          const inputBase = ic * inputArea;
          const weightBase = (oc * inChannels + ic) * kernelArea;
          for (let ky = 0; ky < kernelSize; ky += 1) {
            const iy = oy + ky - padding;
            if (iy < 0 || iy >= height) continue;
            const inputRow = inputBase + iy * width;
            const weightRow = weightBase + ky * kernelSize;
            for (let kx = 0; kx < kernelSize; kx += 1) {
              const ix = ox + kx - padding;
              if (ix < 0 || ix >= width) continue;
              sum += input[inputRow + ix] * layer.weight[weightRow + kx];
            }
          }
        }
        output[outputBase + oy * outWidth + ox] = sum;
      }
    }
  }
  return output;
}

function avgPool2x2(input, channels, height, width) {
  const outHeight = Math.floor(height / 2);
  const outWidth = Math.floor(width / 2);
  const output = new Float32Array(channels * outHeight * outWidth);
  const inputArea = height * width;
  const outPlaneSize = outHeight * outWidth;

  for (let channel = 0; channel < channels; channel += 1) {
    const inputBase = channel * inputArea;
    const outputBase = channel * outPlaneSize;
    for (let y = 0; y < outHeight; y += 1) {
      const inputY = y * 2;
      for (let x = 0; x < outWidth; x += 1) {
        const inputX = x * 2;
        const topLeft = inputBase + inputY * width + inputX;
        output[outputBase + y * outWidth + x] =
          (input[topLeft] +
            input[topLeft + 1] +
            input[topLeft + width] +
            input[topLeft + width + 1]) *
          0.25;
      }
    }
  }
  return output;
}

function dense(input, layer) {
  const output = new Float32Array(layer.outFeatures);
  for (let outIndex = 0; outIndex < layer.outFeatures; outIndex += 1) {
    let sum = layer.bias[outIndex];
    const weightBase = outIndex * layer.inFeatures;
    for (let inIndex = 0; inIndex < layer.inFeatures; inIndex += 1) {
      sum += input[inIndex] * layer.weight[weightBase + inIndex];
    }
    output[outIndex] = sum;
  }
  return output;
}

function tanhArray(values) {
  const output = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) output[i] = Math.tanh(values[i]);
  return output;
}

function softmax(logits) {
  const output = new Float32Array(logits.length);
  let maxValue = -Infinity;
  for (let i = 0; i < logits.length; i += 1)
    if (logits[i] > maxValue) maxValue = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const v = Math.exp(logits[i] - maxValue);
    output[i] = v;
    sum += v;
  }
  for (let i = 0; i < output.length; i += 1) output[i] /= sum;
  return output;
}

function argMax(values) {
  let winner = 0,
    best = values[0];
  for (let i = 1; i < values.length; i += 1)
    if (values[i] > best) {
      best = values[i];
      winner = i;
    }
  return winner;
}

function renderPrediction(probabilities, bestDigit) {
  const confidence = probabilities[bestDigit] * 100;
  predictionDigit.textContent = String(bestDigit);
  predictionDigit.classList.add("has-value");
  predictionConfidence.textContent = `${confidence.toFixed(1)}% confidence`;
  predictionChip.textContent =
    confidence >= 50 ? `Most likely ${bestDigit}` : "Model is unsure";
  updateProbabilityDisplay(probabilities, bestDigit);
}

function updateProbabilityDisplay(probabilities, bestDigit) {
  probabilityRows.forEach(({ row, fill, value }, digit) => {
    const probability = probabilities[digit] * 100;
    fill.style.width = `${probability.toFixed(2)}%`;
    value.textContent = `${probability.toFixed(1)}%`;
    row.classList.toggle("is-top", digit === bestDigit);
  });
}
