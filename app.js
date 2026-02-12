/**
 * Neural Network Design: The Gradient Puzzle
 * FULL WORKING VERSION
 * Noise -> Smooth directional gradient
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.03,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// LOSS HELPERS
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Differentiable alternative to Sorted MSE
// keeps histogram / distribution of colors
function conservationLoss(yTrue, yPred) {
  const trueFlat = yTrue.reshape([-1]);
  const predFlat = yPred.reshape([-1]);

  const trueMean = tf.mean(trueFlat);
  const predMean = tf.mean(predFlat);
  const meanLoss = tf.square(trueMean.sub(predMean));

  const trueVar = tf.mean(tf.square(trueFlat.sub(trueMean)));
  const predVar = tf.mean(tf.square(predFlat.sub(predMean)));
  const varLoss = tf.square(trueVar.sub(predVar));

  return meanLoss.add(varLoss);
}

// Smoothness (Total Variation)
function smoothness(yPred) {
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));

  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));

  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

// Direction: dark-left -> bright-right
function directionX(yPred) {
  const mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// MODELS
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
  }

  if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// STUDENT LOSS
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {

    // conserve color distribution
    const lossConserve = conservationLoss(yTrue, yPred).mul(1.0);

    // smooth neighbors
    const lossSmooth = smoothness(yPred).mul(0.4);

    // force gradient direction
    const lossDir = directionX(yPred).mul(0.8);

    return lossConserve.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// TRAINING STEP
// ==========================================

async function trainStep() {
  state.step++;

  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const pred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, pred);
    }, state.baselineModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  const studentLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const pred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, pred);
    }, state.studentModel.getWeights());

    state.optimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  log(
    `Step ${state.step} | Base=${baselineLossVal.toFixed(4)} | Student=${studentLossVal.toFixed(4)}`
  );

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// INIT + UI
// ==========================================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  resetModels();

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );

  document.getElementById("btn-train").addEventListener("click", trainStep);
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Press Auto Train.");
}

function resetModels(archType = "transformation") {
  if (state.baselineModel) state.baselineModel.dispose();
  if (state.studentModel) state.studentModel.dispose();
  if (state.optimizer) state.optimizer.dispose();

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(archType);
  state.optimizer = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  render();
}

// ==========================================
// RENDER
// ==========================================

async function render() {
  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));

  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText =
    "Loss: " + base.toFixed(5);
  document.getElementById("loss-student").innerText =
    "Loss: " + stud.toFixed(5);
}

function log(msg) {
  const el = document.getElementById("log-area");
  const span = document.createElement("div");
  span.innerText = "> " + msg;
  el.prepend(span);
}

// ==========================================
// AUTO TRAIN
// ==========================================

function toggleAutoTrain() {
  state.isAutoTraining = !state.isAutoTraining;
  loop();
}

function loop() {
  if (state.isAutoTraining) {
    trainStep();
    setTimeout(loop, CONFIG.autoTrainSpeed);
  }
}

init();
