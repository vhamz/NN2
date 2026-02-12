/**
 * Neural Network Design: The Gradient Puzzle
 */

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
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
// Helper Functions (ALL fully differentiable)
// ==========================================

function mse(yTrue, yPred) {
  return tf.losses.meanSquaredError(yTrue, yPred);
}

// Distribution conservation: match mean and variance of pixel values
// This replaces Sorted MSE — same idea ("keep same colors") but differentiable
function conservationLoss(yTrue, yPred) {
  var trueFlat = yTrue.reshape([-1]);
  var predFlat = yPred.reshape([-1]);

  var trueMean = tf.mean(trueFlat);
  var predMean = tf.mean(predFlat);
  var meanLoss = tf.square(trueMean.sub(predMean));

  var trueVar = tf.mean(tf.square(trueFlat.sub(trueMean)));
  var predVar = tf.mean(tf.square(predFlat.sub(predMean)));
  var varLoss = tf.square(trueVar.sub(predVar));

  return meanLoss.add(varLoss);
}

// Smoothness: Total Variation — penalize pixel-to-pixel jumps
function smoothness(yPred) {
  var diffX = yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1])
    .sub(yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]));
  var diffY = yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1])
    .sub(yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]));
  return tf.mean(tf.square(diffX)).add(tf.mean(tf.square(diffY)));
}

// Direction: encourage dark-left, bright-right
function directionX(yPred) {
  var mask = tf.linspace(-1, 1, 16).reshape([1, 1, 16, 1]);
  return tf.mean(yPred.mul(mask)).mul(-1);
}

// ==========================================
// Model Architecture
// ==========================================

function createBaselineModel() {
  var model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

function createStudentModel(archType) {
  var model = tf.sequential();
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
// Custom Loss
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(function() {
    // Conservation: keep the same distribution of pixel values
    var lossConserve = conservationLoss(yTrue, yPred).mul(1.0);
    // Smoothness: be locally consistent
    var lossSmooth = smoothness(yPred).mul(0.3);
    // Direction: dark left, bright right
    var lossDir = directionX(yPred).mul(0.5);
    return lossConserve.add(lossSmooth).add(lossDir);
  });
}

// ==========================================
// Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized.", true);
    stopAutoTrain();
    return;
  }

  var baselineLossVal = tf.tidy(function() {
    var r = tf.variableGrads(function() {
      return mse(state.xInput, state.baselineModel.predict(state.xInput));
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(r.grads);
    return r.value.dataSync()[0];
  });

  var studentLossVal = 0;
  try {
    studentLossVal = tf.tidy(function() {
      var r = tf.variableGrads(function() {
        return studentLoss(state.xInput, state.studentModel.predict(state.xInput));
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(r.grads);
      return r.value.dataSync()[0];
    });
    log("Step " + state.step + ": Base Loss=" + baselineLossVal.toFixed(4) +
        " | Student Loss=" + studentLossVal.toFixed(4));
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
// UI
// ==========================================

function init() {
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);
  resetModels();
  tf.browser.toPixels(state.xInput.squeeze(), document.getElementById("canvas-input"));

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
