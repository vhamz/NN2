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
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50,
};

let state = {
  step: 0,
  isAutoTraining: false,
  autoTrainInterval: null,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizer: null,
};

// ==========================================
// 2. Helper Functions (Loss Components)
// ==========================================

// Standard MSE
function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// -------------------------------------------------------
// Distribution Matching Loss (differentiable Sorted MSE alternative)
// -------------------------------------------------------
// tf.topk has NO gradient in TF.js, so we cannot sort inside training.
// Instead we match the pixel value distribution:
//   - Match mean (conserve average brightness)
//   - Match variance (conserve spread)
//   - Soft histogram matching (conserve distribution shape)
function distributionLoss(yTrue, yPred) {
  const trueFlat = yTrue.reshape([-1]);
  const predFlat = yPred.reshape([-1]);

  // 1) Match mean
  const trueMean = tf.mean(trueFlat);
  const predMean = tf.mean(predFlat);
  const meanLoss = tf.square(trueMean.sub(predMean));

  // 2) Match variance
  const trueVar = tf.mean(tf.square(trueFlat.sub(trueMean)));
  const predVar = tf.mean(tf.square(predFlat.sub(predMean)));
  const varLoss = tf.square(trueVar.sub(predVar));

  // 3) Soft histogram matching — 8 bins with sigmoid gates
  const nBins = 8;
  let histLoss = tf.scalar(0);
  for (let i = 0; i < nBins; i++) {
    const lo = i / nBins;
    const hi = (i + 1) / nBins;
    const trueCount = tf.mean(
      tf.sigmoid(trueFlat.sub(lo).mul(20.0))
        .mul(tf.sigmoid(tf.scalar(hi).sub(trueFlat).mul(20.0)))
    );
    const predCount = tf.mean(
      tf.sigmoid(predFlat.sub(lo).mul(20.0))
        .mul(tf.sigmoid(tf.scalar(hi).sub(predFlat).mul(20.0)))
    );
    histLoss = histLoss.add(tf.square(trueCount.sub(predCount)));
  }

  return meanLoss.add(varLoss).add(histLoss.mul(0.5));
}

// Smoothness (Total Variation Loss)
function smoothness(yPred) {
  const diffX = yPred
    .slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
  const diffY = yPred
    .slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

// Directionality — left-dark, right-bright
function directionX(yPred) {
  const width = 16;
  const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// 3. Model Architecture
// ==========================================

function createBaselineModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// [TODO-A] All three architecture types implemented
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    throw new Error("Unknown architecture type: " + archType);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// [TODO-B] Custom loss: Distribution + Smoothness + Direction
function studentLoss(yTrue, yPred) {
  return tf.tidy(function() {
    var lossDist = distributionLoss(yTrue, yPred);
    var lossSmooth = smoothness(yPred).mul(0.1);
    var lossDir = directionX(yPred).mul(0.1);
    return lossDist.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  var baselineLossVal = tf.tidy(function() {
    var result = tf.variableGrads(function() {
      var yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(result.grads);
    return result.value.dataSync()[0];
  });

  var studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(function() {
      var result = tf.variableGrads(function() {
        var yPred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, yPred);
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(result.grads);
      return result.value.dataSync()[0];
    });
    log("Step " + state.step + ": Base Loss=" + baselineLossVal.toFixed(4) + " | Student Loss=" + studentLossVal.toFixed(4));
  } catch (e) {
    log("Error in Student Training: " + e.message, true);
    stopAutoTrain();
    return;
  }

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baselineLossVal, studentLossVal);
  }
}

// ==========================================
// 6. UI & Initialization
// ==========================================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );

  document.getElementById("btn-train").addEventListener("click", function() { trainStep(); });
  document.getElementById("btn-auto").addEventListener("click", toggleAutoTrain);
  document.getElementById("btn-reset").addEventListener("click", resetModels);

  document.querySelectorAll('input[name="arch"]').forEach(function(radio) {
    radio.addEventListener("change", function(e) {
      resetModels(e.target.value);
      document.getElementById("student-arch-label").innerText =
        e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
    });
  });

  log("Initialized. Ready to train.");
}

function resetModels(archType) {
  if (typeof archType !== "string") archType = null;
  if (state.isAutoTraining) stopAutoTrain();

  if (!archType) {
    var checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "transformation";
  }

  if (state.baselineModel) { state.baselineModel.dispose(); state.baselineModel = null; }
  if (state.studentModel) { state.studentModel.dispose(); state.studentModel = null; }
  if (state.optimizer) { state.optimizer.dispose(); state.optimizer = null; }

  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log("Error creating model: " + e.message, true);
    state.studentModel = createBaselineModel();
  }

  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;
  log("Models reset. Student Arch: " + archType);
  render();
}

async function render() {
  var basePred = state.baselineModel.predict(state.xInput);
  var studPred = state.studentModel.predict(state.xInput);
  await tf.browser.toPixels(basePred.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(studPred.squeeze(), document.getElementById("canvas-student"));
  basePred.dispose();
  studPred.dispose();
}

function updateLossDisplay(base, stud) {
  document.getElementById("loss-baseline").innerText = "Loss: " + base.toFixed(5);
  document.getElementById("loss-student").innerText = "Loss: " + stud.toFixed(5);
}

function log(msg, isError) {
  var el = document.getElementById("log-area");
  var span = document.createElement("div");
  span.innerText = "> " + msg;
  if (isError) span.classList.add("error");
  el.prepend(span);
}

function toggleAutoTrain() {
  var btn = document.getElementById("btn-auto");
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
  var btn = document.getElementById("btn-auto");
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

init();
