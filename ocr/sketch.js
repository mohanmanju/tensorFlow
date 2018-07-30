let model;
let input;
let clr;
let train;
let frame;

let WIDTH = 280;
let HEIGHT = 280;

function preload() {
  //initilizing the model
  model = new Classifier();

}

function setup() {

  frame = createCanvas(WIDTH, HEIGHT);
  background(0);
  input = createButton("Guess");
  input.mousePressed(guess);

  clr = createButton("Clear");
  clr.mousePressed(clearCanvas);

}

function draw() {
  stroke(255);
  strokeWeight(3);
  if (mouseIsPressed) {
    line(mouseX, mouseY, pmouseX, pmouseY);
  }
}

function guess() {
  //let temp = get(frame)
  loadPixels();
  getGoodPixels();

}

function clearCanvas(){
  frame.background(0);
}
function getGoodPixels() {
  let data = [];
  for (let i = 0; i < WIDTH * HEIGHT *4 ; i += 400) {
  data.push(pixels[i]%254);
    //if (pixels[i] > 50) {
      //data.push(1);
//    }else{
  //    data.push(0);
    //}
  }
  //data.push(0);
  console.log(data);
  console.log(data.length);
  data = tf.tensor4d(data,[1,28,28,1]);
  model.evaluate(data);

}

const change = (evt) => {

  let temp = evt.target.innerHTML;

  //toggles values of inout buttons
  if (temp == "0") {
    evt.target.innerHTML = "1"
  } else {
    evt.target.innerHTML = "0"
  }

}

const predict = () => {

  let temp1 = parseInt(input_x.elt.innerHTML);
  let temp2 = parseInt(input_y.elt.innerHTML);
  data = tf.tensor2d([[temp1, temp2]]);

  //gets xor using the model
  model.evaluate(data);

}


