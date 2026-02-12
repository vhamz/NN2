/**
 * Neural Network Design: The Gradient Puzzle
 * All loss computations use only basic tf ops: add, sub, mul, square, mean, slice, reshape
 * No tf.losses.*, no tf.topk, no tf.moments, no tf.scalar inside training
 */

var CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.05,
  autoTrainSpeed: 50
};

var state = {
  step: 0,
  isAutoTraining: false,
  xInput: null,
  baselineModel: null,
  studentModel: null,
  optimizer: null
};

// Pre-computed direction mask (created once, reused)
// Shape [1, 1, 16, 1] — values from -1 to +1 across width
var dirMask = null;
var negOne = null;
var weight03 = null;
var weight05 = null;

function createConstants() {
  var vals = [];
  for (var i = 0; i < 16; i++) {
    vals.push(-1 + (2 * i) / 15);
  }
  dirMask = tf.tensor4d([vals.map(function(v) { return [v]; })], [1, 1, 16, 1]);
  negOne = tf.tensor1d([-1]).reshape([]);
  weight03 = tf.tensor1d([0.3]).reshape([]);
  weight05 = tf.tensor1d([0.5]).reshape([]);
}

// ==========================================
// LOSS HELPERS — pure basic ops only
// ==========================================

// MSE by hand: mean of squared differences
function manualMSE(a, b) {
  var diff = a.sub(b);
  return tf.mean(diff.mul(diff));
}

// Conservation: force output to have same mean & variance as input
function conservationLoss(yTrue, yPred) {
  var trueFlat = yTrue.reshape([256]);
  var predFlat = yPred.reshape([256]);

  // Mean matching
  var trueMean = tf.mean(trueFlat);
  var predMean = tf.mean(predFlat);
  var diffMean = trueMean.sub(predMean);
  var meanLoss = diffMean.mul(diffMean);

  // Variance matching (var = mean((x - mean)^2))
  var trueCentered = trueFlat.sub(trueMean);
  var predCentered = predFlat.sub(predMean);
  var trueVar = tf.mean(trueCentered.mul(trueCentered));
  var predVar = tf.mean(predCentered.mul(predCentered));
  var diffVar = trueVar.sub(predVar);
  var varLoss = diffVar.mul(diffVar);

  return meanLoss.add(varLoss);
}

// Smoothness: penalize differences between neighboring pixels
function smoothnessLoss(yPred) {
  // Horizontal: difference between pixel[row][col] and pixel[row][col+1]
  var left  = yPred.slice([0, 0, 0, 0], [1, 16, 15, 1]);
  var right = yPred.slice([0, 0, 1, 0], [1, 16, 15, 1]);
  var diffH = left.sub(right);
  var hLoss = tf.mean(diffH.mul(diffH));

  // Vertical: difference between pixel[row][col] and pixel[row+1][col]
  var top    = yPred.slice([0, 0, 0, 0], [1, 15, 16, 1]);
  var bottom = yPred.slice([0, 1, 0, 0], [1, 15, 16, 1]);
  var diffV = top.sub(bottom);
  var vLoss = tf.mean(diffV.mul(diffV));

  return hLoss.add(vLoss);
}

// Direction: encourage bright pixels on the right, dark on the left
// Multiply output by mask [-1...+1] and minimize negative correlation
function directionLoss(yPred) {
  var product = yPred.mul(dirMask);
  // We want to MAXIMIZE this (bright * positive = good)
  // So we MINIMIZE the negative
  return tf.mean(product).mul(negOne);
}

// ==========================================
// MODELS
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
    throw new Error("Unknown architecture: " + archType);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// COMBINED STUDENT LOSS
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(function() {
    var lc = conservationLoss(yTrue, yPred);
    var ls = smoothnessLoss(yPred).mul(weight03);
    var ld = directionLoss(yPred).mul(weight05);
    return lc.add(ls).add(ld);
  });
}

// ==========================================
// TRAINING
// ==========================================

async function trainStep() {
  state.step++;

  if (!state.studentModel) {
    log("Error: no student model", true);
    stopAutoTrain();
    return;
  }

  // Baseline: MSE only
  var baseLoss = tf.tidy(function() {
    var r = tf.variableGrads(function() {
      var pred = state.baselineModel.predict(state.xInput);
      return manualMSE(state.xInput, pred);
    }, state.baselineModel.getWeights());
    state.optimizer.applyGradients(r.grads);
    return r.value.dataSync()[0];
  });

  // Student: custom loss
  var studLoss = 0;
  try {
    studLoss = tf.tidy(function() {
      var r = tf.variableGrads(function() {
        var pred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, pred);
      }, state.studentModel.getWeights());
      state.optimizer.applyGradients(r.grads);
      return r.value.dataSync()[0];
    });
    log("Step " + state.step + ": Base=" + baseLoss.toFixed(4) + " | Student=" + studLoss.toFixed(4));
  } catch (e) {
    log("ERROR: " + e.message, true);
    stopAutoTrain();
    return;
  }

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baseLoss, studLoss);
  }
}

// ==========================================
// UI
// ==========================================

function init() {
  createConstants();
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
    var c = document.querySelector('input[name="arch"]:checked');
    archType = c ? c.value : "transformation";
  }

  if (state.baselineModel) { state.baselineModel.dispose(); state.baselineModel = null; }
  if (state.studentModel) { state.studentModel.dispose(); state.studentModel = null; }
  if (state.optimizer) { state.optimizer.dispose(); state.optimizer = null; }

  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log("Model error: " + e.message, true);
    state.studentModel = createBaselineModel();
  }
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;
  log("Reset. Arch: " + archType);
  render();
}

async function render() {
  var bp = state.baselineModel.predict(state.xInput);
  var sp = state.studentModel.predict(state.xInput);
  await tf.browser.toPixels(bp.squeeze(), document.getElementById("canvas-baseline"));
  await tf.browser.toPixels(sp.squeeze(), document.getElementById("canvas-student"));
  bp.dispose();
  sp.dispose();
}

function updateLossDisplay(b, s) {
  document.getElementById("loss-baseline").innerText = "Loss: " + b.toFixed(5);
  document.getElementById("loss-student").innerText = "Loss: " + s.toFixed(5);
}

function log(msg, isErr) {
  var el = document.getElementById("log-area");
  var d = document.createElement("div");
  d.innerText = "> " + msg;
  if (isErr) d.classList.add("error");
  el.prepend(d);
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
