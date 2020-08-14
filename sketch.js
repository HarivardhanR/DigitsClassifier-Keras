const len = 784;

let pred_xs;

let txt = '';
let predArr = [];

let leftBuffer;
let rightBuffer;

let model;

async function preload() {
  // Loading the model
  model = await tf.loadLayersModel('MNISTv99.53/model.json');
}

function setup() {
  createCanvas(800, 280);
  background(255);

  //creating two canvas
  leftBuffer = createGraphics(280, 280);
  rightBuffer = createGraphics(520, 280);

  predArr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  let predButton = select('#predict');
  predButton.mousePressed(function () {
    let inputs = [];
    let img = get(0, 0, 280, 280);
    img.resize(28, 28);
    img.loadPixels();
    for (let i = 0; i < len; i++) {
      let bright = img.pixels[i * 4];
      inputs[i] = (255 - bright) / 255.0;
    }
    let testing = [];
    testing[0] = inputs;
    let xs = tf.tensor2d(testing);
    pred_xs = xs.reshape([1, 28, 28, 1]);
    predict();
    xs.dispose();
    pred_xs.dispose();
  });

  let clearButton = select('#Clear');
  clearButton.mousePressed(function () {
    background(255);
  });

}


function predict() {
  const preds = model.predict(pred_xs);
  const values = preds.dataSync();
  predArr = Array.from(values);
  let m = max(predArr);
  txt = 'Model predicts the digit is ' + predArr.indexOf(m);
  preds.dispose();
}

function draw() {
  drawLeftBuffer();
  drawRightBuffer();

  // Paint the off-screen buffers onto the main canvas
  image(leftBuffer, 0, 0);
  image(rightBuffer, 280, 0);
}

function drawLeftBuffer() {
  strokeWeight(20);
  stroke(0);
  if (mouseIsPressed) {
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
}

function drawRightBuffer() {
  rightBuffer.background(218, 247, 166);
  rightBuffer.fill(255);
  rightBuffer.textSize(32);
  for (var i = 0, j = 0; i <= 280; i += 28, j++) {
    if (predArr[j] < 0.01) {
      rightBuffer.fill(255);
    } else if (predArr[j] < 0.1) {
      rightBuffer.fill(200);
    } else if (predArr[j] < 0.3) {
      rightBuffer.fill(150);
    } else if (predArr[j] < 0.6) {
      rightBuffer.fill(100);
    } else if (predArr[j] < 0.8) {
      rightBuffer.fill(50);
    } else if (predArr[j] >= 0.8) {
      rightBuffer.fill(0);
    }
    rightBuffer.ellipse(40, 13 + i, 10, 10);
    push();
    rightBuffer.fill(0);
    rightBuffer.textSize(10);
    rightBuffer.text("" + j, 47, 17 + i);
    pop();
  }
  noStroke();
  rightBuffer.fill(65);
  rightBuffer.textSize(18);
  rightBuffer.text(txt, 100, 90);
}