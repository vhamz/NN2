const CONFIG = {
  lr: 0.03,
  shapeModel: [16,16,1],
  shapeData: [1,16,16,1]
};

let state = {
  x: null,
  baseline: null,
  student: null,
  optBase: null,
  optStudent: null,
  auto:false
};

//////////////////////////////////////////////////////////////
// LOSS PART
//////////////////////////////////////////////////////////////

function sortedMSE(a,b){
  const sa = tf.sort(a.flatten());
  const sb = tf.sort(b.flatten());
  return tf.mean(tf.square(sa.sub(sb)));
}

function smoothness(y){
  const dx = y.slice([0,0,0,0],[-1,-1,15,-1])
            .sub(y.slice([0,0,1,0],[-1,-1,15,-1]));

  const dy = y.slice([0,0,0,0],[-1,15,-1,-1])
            .sub(y.slice([0,1,0,0],[-1,15,-1,-1]));

  return tf.mean(tf.square(dx)).add(tf.mean(tf.square(dy)));
}

function direction(y){
  const mask = tf.linspace(-1,1,16).reshape([1,1,16,1]);
  return tf.mean(y.mul(mask)).mul(-1);
}

function studentLoss(x,y){
  return tf.tidy(()=>{
    const l1 = sortedMSE(x,y).mul(1.0);
    const l2 = smoothness(y).mul(0.15);
    const l3 = direction(y).mul(0.25);
    return l1.add(l2).add(l3);
  });
}

//////////////////////////////////////////////////////////////
// MODELS
//////////////////////////////////////////////////////////////

function baselineModel(){
  const m = tf.sequential();
  m.add(tf.layers.flatten({inputShape:CONFIG.shapeModel}));
  m.add(tf.layers.dense({units:64,activation:'relu'}));
  m.add(tf.layers.dense({units:256,activation:'sigmoid'}));
  m.add(tf.layers.reshape({targetShape:[16,16,1]}));
  return m;
}

function studentModel(type){
  const m = tf.sequential();
  m.add(tf.layers.flatten({inputShape:CONFIG.shapeModel}));

  if(type==="compression"){
    m.add(tf.layers.dense({units:64,activation:'relu'}));
    m.add(tf.layers.dense({units:256,activation:'sigmoid'}));
  }

  if(type==="transformation"){
    m.add(tf.layers.dense({units:256,activation:'relu'}));
    m.add(tf.layers.dense({units:256,activation:'relu'}));
    m.add(tf.layers.dense({units:256,activation:'sigmoid'}));
  }

  if(type==="expansion"){
    m.add(tf.layers.dense({units:512,activation:'relu'}));
    m.add(tf.layers.dense({units:512,activation:'relu'}));
    m.add(tf.layers.dense({units:256,activation:'sigmoid'}));
  }

  m.add(tf.layers.reshape({targetShape:[16,16,1]}));
  return m;
}

//////////////////////////////////////////////////////////////
// TRAIN
//////////////////////////////////////////////////////////////

function trainStep(){

  tf.tidy(()=>{

    const baseGrads = tf.variableGrads(()=>{
      const y = state.baseline.predict(state.x);
      return tf.losses.meanSquaredError(state.x,y);
    }, state.baseline.trainableWeights.map(w=>w.val));

    state.optBase.applyGradients(baseGrads.grads);

    const studGrads = tf.variableGrads(()=>{
      const y = state.student.predict(state.x);
      return studentLoss(state.x,y);
    }, state.student.trainableWeights.map(w=>w.val));

    state.optStudent.applyGradients(studGrads.grads);
  });

  render();
}

//////////////////////////////////////////////////////////////
// RENDER
//////////////////////////////////////////////////////////////

async function render(){

  const b = state.baseline.predict(state.x);
  const s = state.student.predict(state.x);

  await tf.browser.toPixels(state.x.squeeze(), document.getElementById("input"));
  await tf.browser.toPixels(b.squeeze(), document.getElementById("baseline"));
  await tf.browser.toPixels(s.squeeze(), document.getElementById("student"));

  b.dispose();
  s.dispose();
}

//////////////////////////////////////////////////////////////
// INIT
//////////////////////////////////////////////////////////////

function resetModels(){

  if(state.baseline) state.baseline.dispose();
  if(state.student) state.student.dispose();

  const arch = document.querySelector('input[name="arch"]:checked').value;

  state.baseline = baselineModel();
  state.student = studentModel(arch);

  state.optBase = tf.train.adam(CONFIG.lr);
  state.optStudent = tf.train.adam(CONFIG.lr);

  render();
}

function toggleAuto(){
  state.auto = !state.auto;
  if(state.auto) loop();
}

function loop(){
  if(!state.auto) return;
  trainStep();
  setTimeout(loop,50);
}

//////////////////////////////////////////////////////////////

state.x = tf.randomUniform(CONFIG.shapeData);
resetModels();
render();
