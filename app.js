/**
 * Neural Network Design: The Gradient Puzzle
 * All loss computations use only basic tf ops
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
  baselineOptimizer: null,
  studentOptimizer: null
};

// Pre-computed constants (created once at init)
var dirMask = null;
var negOne = null;

function createConstants() {
  // Direction mask: -1 on left to +1 on right, shape [1, 1, 16, 1]
  var vals = [];
  for (var i = 0; i < 16; i++) {
    vals.push(-1 + (2 * i) / 15);
  }
  dirMask = tf.tensor4d([vals.map(function(v) { return [v]; })], [1, 1, 16, 1]);
  negOne = tf.tensor1d([-1]).reshape([]);
}

// ==========================================
// LOSS HELPERS — only basic ops
// ==========================================

// MSE by hand
function manualMSE(a, b) {
  var diff = a.sub(b);
  return tf.mean(diff.mul(diff));
}

// Conservation: same mean & variance as input
function conservationLoss(yTrue, yPred) {
  var trueFlat = yTrue.reshape([256]);
  var predFlat = yPred.reshape([256]);

  var trueMean = tf.mean(trueFlat);
  var predMean = tf.mean(predFlat);
  var diffMean = trueMean.sub(predMean);
  var meanLoss = diffMean.mul(diffMean);

  var trueCentered = trueFlat.sub(trueMean);
  var predCentered = predFlat.sub(predMean);
  var trueVar = tf.mean(trueCentered.mul(trueCentered));
  var predVar = tf.mean(predCentered.mul(predCentered));
  var diffVar = trueVar.sub(predVar);
  var varLoss = diffVar.mul(diffVar);

  return meanLoss.add(varLoss);
}

// Smoothness: penalize pixel-to-pixel jumps (Total Variation)
function smoothnessLoss(yPred) {
  var left  = yPred.slice([0, 0, 0, 0], [1, 16, 15, 1]);
  var right = yPred.slice([0, 0, 1, 0], [1, 16, 15, 1]);
  var diffH = left.sub(right);
  var hLoss = tf.mean(diffH.mul(diffH));

  var top    = yPred.slice([0, 0, 0, 0], [1, 15, 16, 1]);
  var bottom = yPred.slice([0, 1, 0, 0], [1, 15, 16, 1]);
  var diffV = top.sub(bottom);
  var vLoss = tf.mean(diffV.mul(diffV));

  return hLoss.add(vLoss);
}

// Direction: dark left, bright right
function directionLoss(yPred) {
  var product = yPred.mul(dirMask);
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
    // Bottleneck 256->64->256: loses information, can't rearrange well
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // 1:1 mapping 256->256->256: enough capacity to rearrange all pixels
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // Overcomplete 256->512->256: extra capacity
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
// L = conservation * 2.0 + smoothness * 1.0 + direction * 0.3
// ==========================================

function studentLoss(yTrue, yPred) {
  return tf.tidy(function() {
    // Conservation: keep the same pixel value distribution
    var lc = conservationLoss(yTrue, yPred);

    // Smoothness: be locally consistent (remove noise)
    var ls = smoothnessLoss(yPred);

    // Direction: dark on left, bright on right
    var ld = directionLoss(yPred);

    // Weighted sum: conservation is strong to prevent all-white/all-black
    // smoothness is medium for gradual transitions
    // direction is gentle nudge
    return lc.mul(tf.tensor1d([2.0]).reshape([])).add(
      ls.mul(tf.tensor1d([1.0]).reshape([]))
    ).add(
      ld.mul(tf.tensor1d([0.3]).reshape([]))
    );
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

  // Baseline: MSE only (separate optimizer)
  var baseLoss = tf.tidy(function() {
    var r = tf.variableGrads(function() {
      var pred = state.baselineModel.predict(state.xInput);
      return manualMSE(state.xInput, pred);
    }, state.baselineModel.getWeights());
    state.baselineOptimizer.applyGradients(r.grads);
    return r.value.dataSync()[0];
  });

  // Student: custom loss (separate optimizer)
  var studLoss = 0;
  try {
    studLoss = tf.tidy(function() {
      var r = tf.variableGrads(function() {
        var pred = state.studentModel.predict(state.xInput);
        return studentLoss(state.xInput, pred);
      }, state.studentModel.getWeights());
      state.studentOptimizer.applyGradients(r.grads);
      return r.value.dataSync()[0];
    });
    log("Step " + state.step + " | Base=" + baseLoss.toFixed(4) + " | Student=" + studLoss.toFixed(4));
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

  log("Initialized. Select Transformation arch and click Auto Train.");
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
  if (state.baselineOptimizer) { state.baselineOptimizer.dispose(); state.baselineOptimizer = null; }
  if (state.studentOptimizer) { state.studentOptimizer.dispose(); state.studentOptimizer = null; }

  state.baselineModel = createBaselineModel();
  try {
    state.studentModel = createStudentModel(archType);
  } catch (e) {
    log("Model error: " + e.message, true);
    state.studentModel = createBaselineModel();
  }

  // Separate optimizers so they don't share state between models
  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);
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
