let model;
let input_x;
let intput_y;
let train;

function preload() {
  //initilizing the model
  model = new Classifier();
  
}

function setup() {

  //for inputing values
  input_x = createButton("0");
  input_y = createButton("0");

  input_x.mousePressed(change);
  input_y.mousePressed(change);

  solve = createButton("xor  = >");
  solve.mousePressed(predict);

  result = createButton("-");

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


