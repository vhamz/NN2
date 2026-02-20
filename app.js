// app.js - The Gradient Puzzle (FINAL WORKING VERSION)
// Uses model.fit() - no manual gradients, no errors

// --- Global state ---
let xInput;
let targetRamp;
let baselineModel;
let studentModel;
let stepCount = 0;
let autoTraining = false;
let currentArch = 'compression';

const LAMBDA_TV = 0.1;
const LAMBDA_DIR = 0.01;

// --- Utility functions ---
function log(message, isError = false) {
    const logDiv = document.getElementById('logArea');
    const className = isError ? 'error' : 'info';
    logDiv.innerHTML += `<span class="${className}">> ${message}</span><br>`;
    logDiv.scrollTop = logDiv.scrollHeight;
}

function clearLog() {
    document.getElementById('logArea').innerHTML = '';
}

function createTargetRamp() {
    return tf.tidy(() => {
        const colVals = tf.linspace(0, 1, 16);
        const rows = tf.ones([16, 1]).mul(colVals);
        return rows.reshape([1, 16, 16, 1]);
    });
}

function createFixedNoise() {
    return tf.randomUniform([1, 16, 16, 1], 0, 1);
}

// --- Custom student loss for model.compile ---
const studentLossFn = (yTrue, yPred) => {
    const mseLoss = tf.losses.meanSquaredError(yTrue, yPred);
    
    // Total variation smoothness
    const rightDiff = yPred.slice([0,0,0,0], [1,16,15,1]).sub(yPred.slice([0,0,1,0], [1,16,15,1]));
    const downDiff = yPred.slice([0,0,0,0], [1,15,16,1]).sub(yPred.slice([0,1,0,0], [1,15,16,1]));
    const tvLoss = tf.square(rightDiff).sum().add(tf.square(downDiff).sum());
    
    // Direction alignment
    const dirLoss = yPred.mul(targetRamp).mean().neg();
    
    return mseLoss.add(tf.scalar(LAMBDA_TV).mul(tvLoss)).add(tf.scalar(LAMBDA_DIR).mul(dirLoss));
};

// --- Model creators with compile ---
function createBaselineModel() {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });
    return model;
}

function createStudentModel(archType) {
    const model = tf.sequential();
    model.add(tf.layers.flatten({ inputShape: [16, 16, 1] }));

    if (archType === 'compression') {
        model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    } else if (archType === 'transformation') {
        model.add(tf.layers.dense({ units: 256, activation: 'relu' }));
    } else if (archType === 'expansion') {
        model.add(tf.layers.dense({ units: 512, activation: 'relu' }));
    }

    model.add(tf.layers.dense({ units: 256, activation: 'sigmoid' }));
    model.add(tf.layers.reshape({ targetShape: [16, 16, 1] }));
    model.compile({ optimizer: tf.train.adam(0.01), loss: studentLossFn });
    return model;
}

// --- Training step using model.fit (GUARANTEED TO WORK) ---
async function trainStep() {
    try {
        // Single step training (1 epoch = 1 step for batchSize=1)
        await baselineModel.fit(xInput, targetRamp, {
            epochs: 1,
            batchSize: 1,
            verbose: 0
        });

        await studentModel.fit(xInput, targetRamp, {
            epochs: 1,
            batchSize: 1,
            verbose: 0
        });

        stepCount++;

        // Update display and log MSE for fair comparison
        const predBaseline = baselineModel.predict(xInput);
        const predStudent = studentModel.predict(xInput);
        const mseBaseline = tf.losses.meanSquaredError(targetRamp, predBaseline).dataSync()[0];
        const mseStudent = tf.losses.meanSquaredError(targetRamp, predStudent).dataSync()[0];

        log(`Step ${stepCount} | Baseline MSE: ${mseBaseline.toFixed(4)} | Student MSE: ${mseStudent.toFixed(4)}`);
        updateCanvases(predBaseline, predStudent);

    } catch (e) {
        log(`Error: ${e.message}`, true);
        console.error(e);
        stopAutoTrain();
    }
}

// --- Canvas rendering ---
function updateCanvases(predBaseline, predStudent) {
    const inputData = xInput.dataSync();
    drawCanvas('canvasInput', inputData);
    
    const baseData = predBaseline.dataSync();
    drawCanvas('canvasBaseline', baseData);
    
    const studentData = predStudent.dataSync();
    drawCanvas('canvasStudent', studentData);
    
    predBaseline.dispose();
    predStudent.dispose();
}

function drawCanvas(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(16, 16);
    for (let i = 0; i < 256; i++) {
        const val = Math.floor(data[i] * 255);
        imageData.data[i*4] = val;
        imageData.data[i*4+1] = val;
        imageData.data[i*4+2] = val;
        imageData.data[i*4+3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

// --- Reset ---
async function resetModels() {
    stopAutoTrain();
    
    baselineModel?.dispose();
    studentModel?.dispose();
    
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();
    
    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);
    stepCount = 0;

    const predBase = baselineModel.predict(xInput);
    const predStudent = studentModel.predict(xInput);
    updateCanvases(predBase, predStudent);

    clearLog();
    log('Models reset. Ready.');
}

// --- Auto training ---
let autoInterval;
function startAutoTrain() {
    autoTraining = true;
    document.getElementById('autoTrain').textContent = '⏸ Stop';
    autoInterval = setInterval(trainStep, 100);
}

function stopAutoTrain() {
    autoTraining = false;
    document.getElementById('autoTrain').textContent = '▶ Auto Train';
    if (autoInterval) {
        clearInterval(autoInterval);
        autoInterval = null;
    }
}

// --- Initialization ---
async function init() {
    await tf.ready();
    
    xInput = createFixedNoise();
    targetRamp = createTargetRamp();

    baselineModel = createBaselineModel();
    studentModel = createStudentModel(currentArch);

    const predBase = baselineModel.predict(xInput);
    const predStudent = studentModel.predict(xInput);
    updateCanvases(predBase, predStudent);

    log('✅ Models ready! Student uses custom smoothness+direction loss.');

    // Event listeners
    document.getElementById('trainStep').addEventListener('click', () => {
        if (autoTraining) stopAutoTrain();
        trainStep();
    });

    document.getElementById('autoTrain').addEventListener('click', () => {
        if (autoTraining) stopAutoTrain();
        else startAutoTrain();
    });

    document.getElementById('reset').addEventListener('click', resetModels);

    document.querySelectorAll('input[name="arch"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            stopAutoTrain();
            currentArch = e.target.value;
            studentModel.dispose();
            studentModel = createStudentModel(currentArch);
            log(`Switched to ${currentArch} architecture`);
            const predStudent = studentModel.predict(xInput);
            const predBase = baselineModel.predict(xInput);
            updateCanvases(predBase, predStudent);
        });
    });
}

window.addEventListener('load', init);


