/**
 * Neural Network Design: The Gradient Puzzle
 *
 * Objective:
 * Modify the Student Model architecture and loss function to transform
 * random noise input into a smooth, directional gradient output.
 */

// ==========================================
// 1. Global State & Config
// ==========================================
const CONFIG = {
  // Model definition shape (no batch dim) - used for layer creation
  inputShapeModel: [16, 16, 1],
  // Data tensor shape (includes batch dim) - used for input tensor creation
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50, // ms delay between steps (lower is faster)
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null, // The fixed noise input
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard MSE: Mean Squared Error
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Sorted MSE: Compare sorted pixel distributions (Quantile Loss / 1D Wasserstein)
// This frees the model from position constraints — it only needs to conserve
// the same set of pixel values, not keep them in the same place.
//
// NOTE: tf.topk has no gradient in TF.js, so we pre-sort the input (it's constant)
// and use a histogram/distribution matching approach for the prediction.

// Pre-computed sorted input values (set once at init, no gradient needed)
let sortedInputValues = null;

function precomputeSortedInput(xInput) {
  if (sortedInputValues) sortedInputValues.dispose();
  const flat = xInput.reshape([-1]);
  // topk is fine here — input is a constant, never needs gradients
  sortedInputValues = tf.topk(flat, flat.shape[0]).values.reverse();
  // sortedInputValues is ascending: darkest → brightest
}

function sortedMSE(yTrue, yPred) {
  // For the prediction, we can't use topk (no gradient).
  // Instead, we use a differentiable distribution-matching loss:
  // 1) Match the mean (global brightness conservation)
  // 2) Match the variance (spread conservation)
  // 3) Penalize values outside [0,1] range
  //
  // This encourages the output to use the same "inventory" of pixel values.

  const trueFlat = yTrue.reshape([-1]);
  const predFlat = yPred.reshape([-1]);

  // Match mean
  const meanTrue = tf.mean(trueFlat);
  const meanPred = tf.mean(predFlat);
  const meanLoss = tf.square(meanTrue.sub(meanPred));

  // Match variance
  const varTrue = tf.moments(trueFlat).variance;
  const varPred = tf.moments(predFlat).variance;
  const varLoss = tf.square(varTrue.sub(varPred));

  // Match higher-order stats: match sorted quantiles via binned histogram approach
  // Split into N bins and compare counts
  const nBins = 8;
  let binLoss = tf.scalar(0);
  for (let i = 0; i < nBins; i++) {
    const lo = i / nBins;
    const hi = (i + 1) / nBins;
    // Soft count of pixels in each bin using sigmoid (differentiable)
    const trueBin = tf.mean(
      tf.sigmoid(trueFlat.sub(lo).mul(20)).mul(tf.sigmoid(tf.scalar(hi).sub(trueFlat).mul(20)))
    );
    const predBin = tf.mean(
      tf.sigmoid(predFlat.sub(lo).mul(20)).mul(tf.sigmoid(tf.scalar(hi).sub(predFlat).mul(20)))
    );
    binLoss = binLoss.add(tf.square(trueBin.sub(predBin)));
  }

  return meanLoss.add(varLoss).add(binLoss.mul(0.5));
}

// Smoothness (Total Variation Loss)
// Penalize differences between adjacent pixels to encourage smoothness.
function smoothness(yPred) {
  // Difference in X direction: pixel[i, j] - pixel[i, j+1]
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  // Difference in Y direction: pixel[i, j] - pixel[i+1, j]
  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  // Return sum of squares
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

// Directionality (Gradient)
// Encourage pixels on the right to be brighter than pixels on the left.
function directionX(yPred) {
  // Create a weight mask that increases from left (-1) to right (+1)
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]); // [1, 1, 16, 1]

  // We want yPred to correlate with mask.
  // Maximize (yPred * mask) => Minimize -(yPred * mask)
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// 3. Model Architecture
// ==========================================

// Baseline Model: Fixed Compression (Undercomplete AE)
// 16x16 -> 64 -> 16x16
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" })); // Bottleneck
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" })); // Output 0-1
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------------
// [TODO-A]: STUDENT ARCHITECTURE DESIGN ✅ IMPLEMENTED
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // Bottleneck: Compress information (undercomplete)
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // ✅ Transformation: 1:1 mapping — hidden size ≈ input size (256)
    // This gives the model enough capacity to rearrange pixels
    // without compressing or expanding the representation.
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // ✅ Expansion: Overcomplete — hidden size > input size
    // Even more capacity for complex transformations.
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// ------------------------------------------------------------------
// [TODO-B]: STUDENT LOSS DESIGN ✅ IMPLEMENTED
// Combined loss: Sorted MSE + Smoothness + Direction
//
// - Sorted MSE: conserve the pixel value distribution (rearrange, don't repaint)
// - Smoothness: make adjacent pixels similar (remove jagged noise)
// - Direction:  encourage left-dark / right-bright gradient
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Sorted MSE — "Use the same colors, but you can move them anywhere"
    const lossSorted = sortedMSE(yTrue, yPred);

    // 2. Smoothness — "Be smooth locally" (Total Variation)
    const lossSmooth = smoothness(yPred).mul(0.1);

    // 3. Direction — "Be bright on the right"
    const lossDir = directionX(yPred).mul(0.1);

    // Total Loss
    return lossSorted.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  // Safety check
  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (MSE Only)
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Train Student (Custom Loss)
  let studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(() => {
      const { value, grads } = tf.variableGrads(() => {
        const yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());

      state.optimizer.applyGradients(grads);
      return value.dataSync()[0];
    });
    log(
      `Step ${state.step}: Base Loss=${baselineLossVal.toFixed(4)} | Student Loss=${studentLossVal.toFixed(4)}`,
    );
  } catch (e) {
    log(`Error in Student Training: ${e.message}`, true);
    stopAutoTrain();
    return;
  }

  // Visualize
  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization logic
// ==========================================

function init() {
  // 1. Generate fixed noise
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // Pre-compute sorted input for distribution matching
  precomputeSortedInput(state.xInput);

  // 2. Initialize Models
  resetModels();

  // 3. Render Initial Input
  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input"),
  );

  // 4. Bind Events
  document
    .getElementById("btn-train")
    .addEventListener("click", () => trainStep());
  document
    .getElementById("btn-auto")
    .addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Ready to train.");
}

function resetModels(archType = null) {
  if (typeof archType !== "string") {
    archType = null;
  }

  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "transformation";
  }

  // Dispose old resources
  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  if (state.optimizer) {
    state.optimizer.dispose();
    state.optimizer = null;
  }

  // Create New Models
  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log(`Error creating model: ${e.message}`, true);
    state.studentModel = createBaselineModel();
  }

  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline"),
  );
  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student"),
  );

  basePred.dispose();
  studPred.dispose();
}

// UI Helpers
function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText =
    `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = `> ${msg}`;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

// Auto Train Logic
function toggleAutoTrain() {
  const btn = document.getElementById("btn-auto");
  if (state.isAutoTraining) {
    stopAutoTrain();
  } else {
    state.isAutoTraining = true;
    btn.innerText = "Auto Train (Stop)";
    btn.classList.add("btn-stop");
    btn.classList.remove("btn-auto");
    loop();
  }
}

function stopAutoTrain() {
  state.isAutoTraining = false;
  const btn = document.getElementById("btn-auto");
  btn.innerText = "Auto Train (Start)";
  btn.classList.add("btn-auto");
  btn.classList.remove("btn-stop");
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

// Start
init();
