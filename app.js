/**
 * Neural Network Design: The Gradient Puzzle (WORKING)
 *
 * Student goal:
 * - Do NOT match pixels position-wise.
 * - Keep the same "inventory" of pixel values (histogram constraint).
 * - Rearrange into a smooth left→right gradient.
 *
 * Loss:
 *   L = sortedMSE(input, output) + λ_tv * TV(output) + λ_dir * Direction(output) + λ_mono * Monotonic(output)
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.02,     // stable + fast enough
  autoTrainSpeed: 40,     // ms
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  baselineOptimizer: null,
  studentOptimizer: null,
};

// ===============================
// Loss components
// ===============================
function mse(a, b) {
  // tf.losses.meanSquaredError returns tensor
  return tf.mean(tf.square(a.sub(b)));
}

/**
 * Histogram / inventory constraint:
 * compare sorted pixel lists (position-free).
 */
function sortedMSE(a, b) {
  return tf.tidy(() => {
    const sa = a.flatten().sort();
    const sb = b.flatten().sort();
    return tf.mean(tf.square(sa.sub(sb)));
  });
}

/**
 * Total variation smoothness:
 * penalize local changes.
 */
function totalVariation(y) {
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
 * Direction: brighter on the right
 * maximize mean(y * mask) => minimize negative
 */
const DIR_MASK = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
function directionX(y) {
  return tf.tidy(() => tf.neg(tf.mean(y.mul(DIR_MASK))));
}

/**
 * Monotonicity: column means should increase left→right.
 * Penalize negative diffs.
 */
function monotonicX(y) {
  return tf.tidy(() => {
    const colMeans = tf.mean(y, [0, 1, 3]); // [16]
    const d = colMeans.slice([1], [15]).sub(colMeans.slice([0], [15])); // [15]
    const neg = tf.relu(tf.neg(d));
    return tf.mean(tf.square(neg));
  });
}

// ===============================
// Student loss
// ===============================
function studentLoss(x, y) {
  return tf.tidy(() => {
    const lSorted = sortedMSE(x, y).mul(1.0);

    // keep TV not too strong (otherwise you get a "flat blob")
    const lTV = totalVariation(y).mul(0.03);

    // these two should be strong to force gradient structure
    const lDir = directionX(y).mul(0.9);
    const lMono = monotonicX(y).mul(0.6);

    return lSorted.add(lTV).add(lDir).add(lMono);
  });
}

// ===============================
// Models
// ===============================
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

// ===============================
// Training (IMPORTANT FIX: use trainableWeights vars)
// ===============================
async function trainStep() {
  state.step++;

  // ---- Baseline (pixel-wise MSE) ----
  const baseLossVal = tf.tidy(() => {
    const varList = state.baselineModel.trainableWeights.map(w => w.val);

    const { value, grads } = tf.variableGrads(() => {
      const y = state.baselineModel.apply(state.xInput, { training: true });
      return mse(state.xInput, y);
    }, varList);

    state.baselineOptimizer.applyGradients(grads);

    // Return a number
    return value.dataSync()[0];
  });

  // ---- Student (custom loss) ----
  const studLossVal = tf.tidy(() => {
    const varList = state.studentModel.trainableWeights.map(w => w.val);

    const { value, grads } = tf.variableGrads(() => {
      const y = state.studentModel.apply(state.xInput, { training: true });
      return studentLoss(state.xInput, y);
    }, varList);

    state.studentOptimizer.applyGradients(grads);

    return value.dataSync()[0];
  });

  log(`Step ${state.step}: Base Loss=${baseLossVal.toFixed(4)} | Student Loss=${studLossVal.toFixed(4)}`);

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baseLossVal, studLossVal);
  }
}

// ===============================
// Render
// ===============================
async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));

  basePred.dispose();
  studPred.dispose();
}

// ===============================
// UI helpers
// ===============================
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
    return;
  }

  state.isAutoTraining = true;
  btn.innerText = "Auto Train (Stop)";
  btn.classList.add("btn-stop");
  btn.classList.remove("btn-auto");
  loop();
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

  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.baselineOptimizer) state.baselineOptimizer.dispose();
  if (state.studentOptimizer) state.studentOptimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);

  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  log(`Models reset. Student Arch: ${archType}`);
  render();
}

function loop() {
  if (!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

// ===============================
// Init
// ===============================
function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

  resetModels();

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

init();
