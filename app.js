/**
 * Neural Network Design: The Gradient Puzzle
 * FINAL WORKING VERSION
 *
 * Задача:
 * Переставить существующие пиксели шума → плавный градиент
 * без создания новых цветов.
 */

// ==========================================
// CONFIG
// ==========================================

const CONFIG = {
  inputShapeModel: [16, 16, 1],
  inputShapeData: [1, 16, 16, 1],
  learningRate: 0.015,
  autoTrainSpeed: 40,
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

// ==========================================
// BASIC LOSS
// ==========================================

function mse(a, b) {
  return tf.losses.meanSquaredError(a, b);
}

// ==========================================
// CORE IDEA — SORTED MSE
// сохраняем histogram, разрешаем перестановку пикселей
// ==========================================

function sortedMSE(a, b) {
  const sa = tf.sort(a.flatten());
  const sb = tf.sort(b.flatten());
  return tf.mean(tf.square(sa.sub(sb)));
}

// ==========================================
// SMOOTHNESS
// ==========================================

function smoothness(y) {

  const dx = y.slice([0,0,0,0],[-1,-1,15,-1])
              .sub(y.slice([0,0,1,0],[-1,-1,15,-1]));

  const dy = y.slice([0,0,0,0],[-1,15,-1,-1])
              .sub(y.slice([0,1,0,0],[-1,15,-1,-1]));

  return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
}

// ==========================================
// DIRECTION (gradient left→right)
// ==========================================

function directionX(y) {
  const mask = tf.linspace(-1, 1, 16).reshape([1,1,16,1]);
  return tf.mean(y.mul(mask)).mul(-1);
}

// ==========================================
// FINAL STUDENT LOSS
// ==========================================

function studentLoss(yTrue, yPred) {

  return tf.tidy(() => {

    // ГЛАВНОЕ — перестановка пикселей
    const lossSorted = sortedMSE(yTrue, yPred).mul(1.0);

    // геометрия
    const lossSmooth = smoothness(yPred).mul(0.08);
    const lossDir = directionX(yPred).mul(0.18);

    return lossSorted
      .add(lossSmooth)
      .add(lossDir);
  });
}

// ==========================================
// MODELS
// ==========================================

function createBaselineModel() {

  const model = tf.sequential();

  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  model.add(tf.layers.reshape({ targetShape: [16,16,1] }));

  return model;
}

function createStudentModel(archType) {

  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {

    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  }

  else if (archType === "transformation") {

    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  }

  else if (archType === "expansion") {

    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));

  }

  model.add(tf.layers.reshape({ targetShape: [16,16,1] }));

  return model;
}

// ==========================================
// TRAIN STEP
// ==========================================

async function trainStep() {

  state.step++;

  // baseline
  const baseLossVal = tf.tidy(() => {

    const { value, grads } = tf.variableGrads(() => {

      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred);

    }, state.baselineModel.trainableWeights.map(w => w.val));

    state.baselineOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  // student
  const studLossVal = tf.tidy(() => {

    const { value, grads } = tf.variableGrads(() => {

      const yPred = state.studentModel.predict(state.xInput);
      return studentLoss(state.xInput, yPred);

    }, state.studentModel.trainableWeights.map(w => w.val));

    state.studentOptimizer.applyGradients(grads);
    return value.dataSync()[0];
  });

  if (state.step % 5 === 0 || !state.isAutoTraining) {
    await render();
    updateLossDisplay(baseLossVal, studLossVal);
  }
}

// ==========================================
// RENDER
// ==========================================

async function render() {

  const basePred = state.baselineModel.predict(state.xInput);
  const studPred = state.studentModel.predict(state.xInput);

  await tf.browser.toPixels(
    basePred.squeeze(),
    document.getElementById("canvas-baseline")
  );

  await tf.browser.toPixels(
    studPred.squeeze(),
    document.getElementById("canvas-student")
  );

  basePred.dispose();
  studPred.dispose();
}

// ==========================================
// UI
// ==========================================

function init() {

  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

  tf.browser.toPixels(
    state.xInput.squeeze(),
    document.getElementById("canvas-input")
  );

  resetModels();

  document.getElementById("btn-train").onclick = trainStep;
  document.getElementById("btn-auto").onclick = toggleAutoTrain;
  document.getElementById("btn-reset").onclick = resetModels;

  document.querySelectorAll('input[name="arch"]').forEach(radio=>{
    radio.onchange = ()=> resetModels();
  });
}

function resetModels() {

  if(state.baselineModel) state.baselineModel.dispose();
  if(state.studentModel) state.studentModel.dispose();

  const arch = document.querySelector('input[name="arch"]:checked').value;

  state.baselineModel = createBaselineModel();
  state.studentModel = createStudentModel(arch);

  state.baselineOptimizer = tf.train.adam(CONFIG.learningRate);
  state.studentOptimizer = tf.train.adam(CONFIG.learningRate);

  state.step = 0;
  render();
}

// ==========================================
// AUTO TRAIN
// ==========================================

function toggleAutoTrain() {
  state.isAutoTraining = !state.isAutoTraining;
  if(state.isAutoTraining) loop();
}

function loop() {
  if(!state.isAutoTraining) return;
  trainStep();
  setTimeout(loop, CONFIG.autoTrainSpeed);
}

// ==========================================

init();
