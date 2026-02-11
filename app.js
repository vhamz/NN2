// ============================================
// GLOBAL STATE
// ============================================
let xInput = null;          // Fixed noise tensor [1, 16, 16, 1]
let baselineModel = null;   // Fixed compression model with MSE
let studentModel = null;    // Student model (editable architecture + loss)
let optimizer = null;       // Adam optimizer
let isTraining = false;     // Auto-train state
let stepCount = 0;          // Training step counter

// DOM Elements
const inputCanvas = document.getElementById('inputCanvas');
const baselineCanvas = document.getElementById('baselineCanvas');
const studentCanvas = document.getElementById('studentCanvas');
const statusLog = document.getElementById('statusLog');
const trainStepBtn = document.getElementById('trainStep');
const autoTrainBtn = document.getElementById('autoTrain');
const resetBtn = document.getElementById('resetWeights');
const archRadios = document.getElementsByName('architecture');

// ============================================
// HELPER FUNCTIONS (Provided for students)
// ============================================

/**
 * Calculate MSE between two tensors
 * @param {tf.Tensor} yTrue - Ground truth
 * @param {tf.Tensor} yPred - Predictions
 * @returns {tf.Tensor} Scalar MSE loss
 */
function mse(yTrue, yPred) {
    return tf.tidy(() => {
        const diff = tf.sub(yTrue, yPred);
        const squared = tf.square(diff);
        return tf.mean(squared);
    });
}

/**
 * Calculate smoothness (Total Variation) loss
 * Penalizes differences between neighboring pixels
 * @param {tf.Tensor} yPred - Predictions [1, 16, 16, 1]
 * @returns {tf.Tensor} Scalar smoothness loss
 */
function smoothness(yPred) {
    return tf.tidy(() => {
        // Horizontal differences
        const horizDiff = tf.sub(
            yPred.slice([0, 0, 1, 0], [-1, -1, 15, -1]),
            yPred.slice([0, 0, 0, 0], [-1, -1, 15, -1])
        );
        
        // Vertical differences
        const vertDiff = tf.sub(
            yPred.slice([0, 1, 0, 0], [-1, 15, -1, -1]),
            yPred.slice([0, 0, 0, 0], [-1, 15, -1, -1])
        );
        
        return tf.add(
            tf.mean(tf.square(horizDiff)),
            tf.mean(tf.square(vertDiff))
        );
    });
}

/**
 * Calculate directional loss (encourage left-dark, right-bright)
 * @param {tf.Tensor} yPred - Predictions [1, 16, 16, 1]
 * @returns {tf.Tensor} Scalar direction loss
 */
function directionX(yPred) {
    return tf.tidy(() => {
        // Create horizontal gradient mask: [-1.0, -0.8, ..., 0.8, 1.0]
        const gradientMask = tf.linspace(-1.0, 1.0, 16)
            .reshape([1, 1, 16, 1])
            .tile([1, 16, 1, 1]);
        
        // Multiply and negate mean (we want right side bright)
        return tf.neg(tf.mean(tf.mul(yPred, gradientMask)));
    });
}

// ============================================
// MODEL CREATION
// ============================================

/**
 * Create baseline model (Compression architecture, MSE only)
 */
function createBaselineModel() {
    const model = tf.sequential();
    
    // Encoder (compression)
    model.add(tf.layers.conv2d({
        inputShape: [16, 16, 1],
        filters: 8,
        kernelSize: 3,
        padding: 'same',
        activation: 'relu'
    }));
    
    model.add(tf.layers.maxPooling2d({
        poolSize: 2
    })); // Now 8x8
    
    // Decoder
    model.add(tf.layers.upSampling2d({
        size: 2
    })); // Back to 16x16
    
    model.add(tf.layers.conv2d({
        filters: 1,
        kernelSize: 3,
        padding: 'same',
        activation: 'sigmoid'
    }));
    
    return model;
}

/**
 * TODO-A: Create student model based on architecture type
 * 
 * @param {string} archType - 'compression' | 'transformation' | 'expansion'
 * @returns {tf.Sequential} TensorFlow.js model
 * 
 * INSTRUCTIONS:
 * - 'compression': IMPLEMENTED (same as baseline above)
 * - 'transformation': NOT IMPLEMENTED - throw error for now
 * - 'expansion': NOT IMPLEMENTED - throw error for now
 * 
 * Transformation should have encoder→bottleneck→decoder with more complexity
 * Expansion should increase dimensions then decode
 */
function createStudentModel(archType) {
    if (archType === 'compression') {
        // IMPLEMENTED: Same as baseline
        return createBaselineModel();
    }
    
    // TODO-A: Implement transformation architecture
    if (archType === 'transformation') {
        throw new Error('TODO-A: Transformation architecture not implemented yet. Please implement createStudentModel() for transformation type.');
    }
    
    // TODO-A: Implement expansion architecture
    if (archType === 'expansion') {
        throw new Error('TODO-A: Expansion architecture not implemented yet. Please implement createStudentModel() for expansion type.');
    }
    
    throw new Error(`Unknown architecture: ${archType}`);
}

// ============================================
// LOSS FUNCTIONS
// ============================================

/**
 * Baseline loss: MSE only
 */
function baselineLoss(yTrue, yPred) {
    return mse(yTrue, yPred);
}

/**
 * TODO-B: Student custom loss
 * 
 * INSTRUCTIONS:
 * Start with MSE, then add smoothness and direction terms with tunable coefficients
 * 
 * Example formula:
 * totalLoss = MSE(yTrue, yPred) + λ1 * smoothness(yPred) + λ2 * directionX(yPred)
 * 
 * Try different λ values (e.g., λ1=0.1, λ2=0.05) and observe the gradient structure
 * 
 * @param {tf.Tensor} yTrue - Ground truth (input)
 * @param {tf.Tensor} yPred - Model predictions
 * @returns {tf.Tensor} Scalar loss value
 */
function studentLoss(yTrue, yPred) {
    // TODO-B: Implement custom loss function
    // For now, just return MSE (same as baseline)
    return mse(yTrue, yPred);
    
    // TODO-B: Uncomment and tune these coefficients:
    // const lambda1 = 0.1;  // Smoothness weight
    // const lambda2 = 0.05; // Direction weight
    // 
    // return tf.tidy(() => {
    //     const mseLoss = mse(yTrue, yPred);
    //     const smoothLoss = smoothness(yPred);
    //     const dirLoss = directionX(yPred);
    //     
    //     return tf.add(
    //         mseLoss,
    //         tf.add(
    //             tf.mul(lambda1, smoothLoss),
    //             tf.mul(lambda2, dirLoss)
    //         )
    //     );
    // });
}

// ============================================
// TRAINING LOOP
// ============================================

/**
 * Perform one training step for a model
 */
async function trainOneStep(model, lossFn) {
    return tf.tidy(() => {
        const loss = optimizer.minimize(() => {
            const predictions = model.predict(xInput);
            return lossFn(xInput, predictions);
        }, true);
        
        return loss;
    });
}

/**
 * Main training function
 */
async function train() {
    stepCount++;
    
    // Train baseline
    const baseLoss = await trainOneStep(baselineModel, baselineLoss);
    const baseLossVal = await baseLoss.data();
    
    // Train student
    const studLoss = await trainOneStep(studentModel, studentLoss);
    const studLossVal = await studLoss.data();
    
    // Update UI
    updateCanvases();
    logStatus(stepCount, baseLossVal[0], studLossVal[0]);
    
    // Cleanup
    baseLoss.dispose();
    studLoss.dispose();
}

// ============================================
// RENDERING
// ============================================

/**
 * Render tensor to canvas (16x16 -> scaled to canvas size)
 */
function renderToCanvas(tensor, canvas) {
    tf.tidy(() => {
        const data = tensor.squeeze().arraySync();
        const ctx = canvas.getContext('2d');
        const pixelSize = canvas.width / 16;
        
        for (let y = 0; y < 16; y++) {
            for (let x = 0; x < 16; x++) {
                const value = Math.floor(data[y][x] * 255);
                ctx.fillStyle = `rgb(${value}, ${value}, ${value})`;
                ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
            }
        }
    });
}

/**
 * Update all canvases with current model outputs
 */
function updateCanvases() {
    tf.tidy(() => {
        // Input (fixed)
        renderToCanvas(xInput, inputCanvas);
        
        // Baseline output
        const basePred = baselineModel.predict(xInput);
        renderToCanvas(basePred, baselineCanvas);
        
        // Baseline loss display
        const baseLoss = baselineLoss(xInput, basePred);
        const baseLossVal = baseLoss.dataSync()[0];
        document.getElementById('baselineLoss').textContent = `Loss: ${baseLossVal.toFixed(4)}`;
        
        // Student output
        const studPred = studentModel.predict(xInput);
        renderToCanvas(studPred, studentCanvas);
        
        // Student loss display
        const studLoss = studentLoss(xInput, studPred);
        const studLossVal = studLoss.dataSync()[0];
        document.getElementById('studentLoss').textContent = `Loss: ${studLossVal.toFixed(4)}`;
    });
}

/**
 * Log training status
 */
function logStatus(step, baseLoss, studLoss, error = null) {
    const statusLine = document.createElement('div');
    statusLine.className = 'status-line';
    
    if (error) {
        statusLine.className += ' error';
        statusLine.textContent = `❌ Error: ${error}`;
    } else {
        statusLine.textContent = `> Step ${step}: Base Loss=${baseLoss.toFixed(4)} | Student Loss=${studLoss.toFixed(4)}`;
    }
    
    statusLog.insertBefore(statusLine, statusLog.firstChild);
    
    // Keep only last 50 lines
    while (statusLog.children.length > 50) {
        statusLog.removeChild(statusLog.lastChild);
    }
}

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize the application
 */
async function init() {
    try {
        // Generate fixed noise input [1, 16, 16, 1]
        xInput = tf.randomUniform([1, 16, 16, 1], 0, 1);
        
        // Create models
        baselineModel = createBaselineModel();
        studentModel = createStudentModel('compression');
        
        // Create optimizer
        optimizer = tf.train.adam(0.01);
        
        // Initial render
        updateCanvases();
        
        console.log('✅ Initialization complete');
    } catch (error) {
        logStatus(0, 0, 0, error.message);
        console.error('Initialization error:', error);
    }
}

// ============================================
// EVENT HANDLERS
// ============================================

trainStepBtn.addEventListener('click', async () => {
    try {
        await train();
    } catch (error) {
        logStatus(stepCount, 0, 0, error.message);
    }
});

autoTrainBtn.addEventListener('click', () => {
    isTraining = !isTraining;
    autoTrainBtn.textContent = isTraining ? 'Auto Train (Stop)' : 'Auto Train (Start)';
    
    if (isTraining) {
        const trainLoop = async () => {
            if (!isTraining) return;
            try {
                await train();
                requestAnimationFrame(trainLoop);
            } catch (error) {
                isTraining = false;
                autoTrainBtn.textContent = 'Auto Train (Start)';
                logStatus(stepCount, 0, 0, error.message);
            }
        };
        trainLoop();
    }
});

resetBtn.addEventListener('click', async () => {
    // Stop training
    isTraining = false;
    autoTrainBtn.textContent = 'Auto Train (Start)';
    
    // Reset step count
    stepCount = 0;
    
    // Properly dispose old models with tf.dispose
    await tf.nextFrame();
    if (baselineModel) {
        tf.dispose(baselineModel);
        baselineModel = null;
    }
    if (studentModel) {
        tf.dispose(studentModel);
        studentModel = null;
    }
    
    // Wait for cleanup
    await tf.nextFrame();
    
    // Recreate models
    try {
        const selectedArch = document.querySelector('input[name="architecture"]:checked').value;
        baselineModel = createBaselineModel();
        studentModel = createStudentModel(selectedArch);
        
        updateCanvases();
        
        const newLog = document.createElement('div');
        newLog.className = 'status-line';
        newLog.textContent = '✅ Weights reset';
        statusLog.insertBefore(newLog, statusLog.firstChild);
        
        console.log('✅ Weights reset');
    } catch (error) {
        logStatus(0, 0, 0, error.message);
    }
});

// Architecture change handler
archRadios.forEach(radio => {
    radio.addEventListener('change', async (e) => {
        const archType = e.target.value;
        document.getElementById('studentArch').textContent = 
            archType.charAt(0).toUpperCase() + archType.slice(1);
        
        // Stop training if active
        isTraining = false;
        autoTrainBtn.textContent = 'Auto Train (Start)';
        
        // Properly dispose student model
        await tf.nextFrame();
        if (studentModel) {
            tf.dispose(studentModel);
            studentModel = null;
        }
        
        await tf.nextFrame();
        
        // Recreate student model
        try {
            studentModel = createStudentModel(archType);
            updateCanvases();
        } catch (error) {
            logStatus(stepCount, 0, 0, error.message);
        }
    });
});

// ============================================
// START APPLICATION
// ============================================
init();
