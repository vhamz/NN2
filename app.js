/**
 * Neural Network Design: The Gradient Puzzle
 * WORKING VERSION (Student: custom loss + proper grads)
 *
 * Goal:
 * - Do NOT copy pixels (no pixel-wise target).
 * - Keep the same "inventory" of pixel values (histogram constraint).
 * - Rearrange them into a smooth left→right gradient.
 *
 * Key:
 *   L = L_sorted + λ_tv * L_tv + λ_dir * L_dir + λ_mono * L_mono
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  // A bit smaller LR is stabler with non-standard losses
  learningRate: 0.015,
  autoTrainSpeed: 40, // ms
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null, // fixed noise
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// ==========================================
// Loss components
// ==========================================
function mse(a, b) {
  return tf.losses.meanSquaredError(a, b);
}

/**
 * Distribution constraint: compare sorted pixels.
 * This approximates "same histogram" and allows free rearrangement.
 */
function sortedMSE(a, b) {
  return tf.tidy(() => {
    const sa = tf.sort(a.flatten()); // ascending
    const sb = tf.sort(b.flatten());
    return tf.mean(tf.square(sa.sub(sb)));
  });
}

/**
 * Total variation (smoothness): penalize local jumps.
 */
function smoothness(y) {
  return tf.tidy(() => {
    const dx = y
      .slice([0, 0, 0, 0], [-1, -1, 15, -1])
      .sub(y.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
    const dy = y
      .slice([0, 0, 0, 0], [-1, 15, -1, -1])
      .sub(y.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
    return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
  });
}

/**
 * Direction: encourage brighter on the right.
 * Maximize mean(y * mask)  => minimize negative.
 */
const DIR_MASK = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]); // [1,1,W,1]
function directionX(y) {
  return tf.tidy(() => tf.neg(tf.mean(y.mul(DIR_MASK))));
}

/**
 * Extra "shape" constraint: enforce monotonic increase along x on average.
 * We take column means and penalize negative differences.
 */
function monotonicX(y) {
  return tf.tidy(() => {
    // y: [1,16,16,1]
    const colMeans = tf.mean(y, [0, 1, 3]); // -> [16]
    const diffs = colMeans.slice([1], [15]).sub(colMeans.slice([0], [15])); // Δ along x
    // penalize only if diffs < 0
    const neg = tf.relu(tf.neg(diffs));
    return tf.mean(tf.square(neg));
  });
}

// ==========================================
// Student loss (THE MAIN FIX)
// ==========================================
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1) conserve pixel values (no new colors)
    const lSorted = sortedMSE(yTrue, yPred).mul(1.0);

    // 2) make it smooth
    const lTV = smoothness(yPred).mul(0.08);

    // 3) enforce left→right direction
    const lDir = directionX(yPred).mul(0.18);

    // 4) (optional but helpful) enforce monotonicity in column means
    const lMono = monotonicX(yPred).mul(0.12);

    return lSorted.add(lTV).add(lDir).add(lMono);
  });
}

// ==========================================
// Models
// ==========================================
function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// Training
// ==========================================
async function trainStep() {
  state.step++;

  // Baseline: pixel-wise MSE (identity)
  const baseLossVal = tf.tidy(() => {
    const vars = state.baselineModel.trainableWeights.map(w => w.val);
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, vars);
    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // Student: custom loss (rearrangement)
  const studLossVal = tf.tidy(() => {
    const vars = state.studentModel.trainableWeights.map(w => w.val);
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, yPred);
    }, vars);
    state.studentOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  log(`Step ${state.step}: Base Loss=${baseLossVal.toFixed(4)} | Student Loss=${studLossVal.toFixed(4)}`);

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baseLossVal, studLossVal);
  }
}

// ==========================================
// Render
// ==========================================
async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));

  basePred.dispose();
  studPred.dispose();
}

// ==========================================
// UI glue (kept same IDs as index.html)
// ==========================================
function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText = `Loss: ${base.toFixed(5)}`;
  document.getElementById("loss-student").innerText = `Loss: ${stud.toFixed(5)}`;
}

function log(msg, isError = false) {
  const el = document.getElementById("log-area");
  const line = document.createElement("div");
  line.innerText = `> ${msg}`;
  if (isError) line.classList.add("error");
  el.prepend(line);
}

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

function resetModels(archType = null) {
  if (typeof archType !== "string") archType = null;
  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "transformation";
  }

  // Dispose old
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineOptimizer) state.baselineOptimizer.dispose();
  if (state.studentOptimizer) state.studentOptimizer.dispose();

  // Recreate
  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  log(`Models reset. Student Arch: ${archType}`);
  render();
}

function init() {
  // Fixed noise
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  // Render input
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

  // Init models
  resetModels();

  // Bind events
  document.getElementById("btn-train").addEventListener("click", () => trainStep());
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", () => resetModels());

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Ready to train.");
}

function loop() {
  if (!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

// Start
init();
