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
  learningRate: 0.1, // Increased from 0.05 to 0.1 for faster learning
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
  return tf.tidy(() => {
    return tf.losses.meanSquaredError(yTrue, yPred);
  });
}

// Sorted MSE: Quantile Loss / Wasserstein Distance
// Allows pixel movement while preserving color distribution
function sortedMSE(yTrue, yPred) {
  return tf.tidy(() => {
    // Flatten and sort both tensors
    const yTrueFlat = yTrue.flatten();
    const yPredFlat = yPred.flatten();
    
    // Sort both arrays
    const yTrueSorted = tf.topk(yTrueFlat, yTrueFlat.shape[0]).values;
    const yPredSorted = tf.topk(yPredFlat, yPredFlat.shape[0]).values;
    
    // MSE on sorted values
    const diff = tf.sub(yTrueSorted, yPredSorted);
    return tf.mean(tf.square(diff));
  });
}

// TODO: Helper - Smoothness (Total Variation)
// Penalize differences between adjacent pixels to encourage smoothness.
function smoothness(yPred) {
  return tf.tidy(() => {
    // Difference in X direction
    const diffX = tf.sub(
      yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1]),
      yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1])
    );

    // Difference in Y direction
    const diffY = tf.sub(
      yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1]),
      yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1])
    );

    return tf.add(tf.mean(tf.square(diffX)), tf.mean(tf.square(diffY)));
  });
}

// TODO: Helper - Directionality (Gradient)
// Encourage pixels on the right to be brighter than pixels on the left.
function directionX(yPred) {
  return tf.tidy(() => {
    const width = 16;
    const mask = tf.linspace(-1, 1, width).reshape([1, 1, width, 1]);
    return tf.mul(-1, tf.mean(tf.mul(yPred, mask)));
  });
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
  // Reshape back to [16, 16, 1] (batch dim is handled automatically)
  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ------------------------------------------------------------------
// [TODO-A]: STUDENT ARCHITECTURE DESIGN
// Modify this function to implement 'transformation' and 'expansion'.
// ------------------------------------------------------------------
function createStudentModel(archType) {
  const model = tf.sequential();
  model.add(tf.layers.flatten({ inputShape: CONFIG.inputShapeModel }));

  if (archType === "compression") {
    // [Implemented] Bottleneck: Compress information
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "transformation") {
    // [IMPLEMENTED] Transformation (1:1 mapping)
    // Maintain dimension - same size as input (256)
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else if (archType === "expansion") {
    // [IMPLEMENTED] Expansion (Overcomplete)
    // Increase dimension - larger hidden layer
    model.add(tf.layers.dense({ units: 512, activation: "relu" }));
    model.add(tf.layers.dense({ units: 256, activation: "sigmoid" }));
  } else {
    // Safety check for unknown architectures
    throw new Error(`Unknown architecture type: ${archType}`);
  }

  model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
  return model;
}

// ==========================================
// 4. Custom Loss Function
// ==========================================

// ------------------------------------------------------------------
// [TODO-B]: STUDENT LOSS DESIGN
// Modify this function to create a smooth gradient.
// Uses Sorted MSE to allow pixel rearrangement.
// ------------------------------------------------------------------
function studentLoss(yTrue, yPred) {
  return tf.tidy(() => {
    // 1. Sorted MSE - Allow pixel movement while preserving colors
    const lossSortedMSE = sortedMSE(yTrue, yPred);

    // 2. Smoothness - Make it VERY smooth (high weight!)
    const lossSmooth = tf.mul(smoothness(yPred), 5.0); // Increased from 0.1 to 5.0

    // 3. Direction - Strong gradient push (high weight!)
    const lossDir = tf.mul(directionX(yPred), 5.0); // Increased from 0.1 to 5.0

    // Total Loss = Sorted MSE + Smoothness + Direction
    return tf.add(tf.add(lossSortedMSE, lossSmooth), lossDir);
  });
}

// ==========================================
// 5. Training Loop
// ==========================================

async function trainStep() {
  state.step++;

  // Safety check: Ensure models are initialized
  if (!state.studentModel || !state.studentModel.getWeights) {
    log("Error: Student model not initialized properly.", true);
    stopAutoTrain();
    return;
  }

  // Train Baseline (MSE Only)
  // We use a simple fit here or gradient tape, let's use tape for consistency
  const baselineLossVal = tf.tidy(() => {
    const { value, grads } = tf.variableGrads(() => {
      const yPred = state.baselineModel.predict(state.xInput);
      return mse(state.xInput, yPred); // Baseline always uses MSE
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
        return studentLoss(state.xInput, yPred); // Uses student's custom loss
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
  // 1. Generate fixed noise (Batch size included: [1, 16, 16, 1])
  state.xInput = tf.randomUniform(CONFIG.inputShapeData);

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
  // [Fix]: When called via event listener, archType is an Event object.
  // We must ensure it's either a string or null.
  if (typeof archType !== "string") {
    archType = null;
  }

  // Safety: Stop auto-training to prevent race conditions during reset
  if (state.isAutoTraining) {
    stopAutoTrain();
  }

  if (!archType) {
    const checked = document.querySelector('input[name="arch"]:checked');
    archType = checked ? checked.value : "compression";
  }

  // Dispose old resources to avoid memory leaks
  if (state.baselineModel) {
    state.baselineModel.dispose();
    state.baselineModel = null;
  }
  if (state.studentModel) {
    state.studentModel.dispose();
    state.studentModel = null;
  }
  // Important: Dispose optimizer because it holds references to old model variables.
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
    state.studentModel = createBaselineModel(); // Fallback to avoid crash
  }

  // Create new optimizer (must be done AFTER models are created)
  state.optimizer = tf.train.adam(CONFIG.learningRate);
  state.step = 0;

  log(`Models reset. Student Arch: ${archType}`);
  render();
}

async function render() {
  // Tensor memory management with tidy not possible here due to async toPixels,
  // so we manually dispose predictions.
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
