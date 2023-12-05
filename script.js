import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js";

// input feature pairs (size, # of bedrooms)
const INPUTS = TRAINING_DATA.inputs;

// current listed house prices in dollars given their features above (target output values you want to predict)
const OUTPUTS = TRAINING_DATA.outputs;

// shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// inputs feature array of arrays needs 2D tensor to store
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// output can stay 1 dimensional
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// function to take a Tensor and normalize values with respect to each column of values contained in that Tensor
function normalize(tensor, min, max) {
  // tf.tidy allows it to monitor any code inside of its function so if any new tensors are created they will be automatically disposed of once the function returns, without needing to directly dispose of tensors myself later on in the code block.
  const result = tf.tidy(function () {
    // find the minium value contained in the Tensor.
    const MIN_VALUES = min || tf.min(tensor, 0);

    // find the maximum value contained in the Tensor
    const MAX_VALUES = max || tf.max(tensor, 0);

    // now subtract the MIN_VALUES from every value in the Tensor and store the results in a new Tensor
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // calculate the range size of possible values
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    //calculate the adjusted values divided by the range size as a new Tensor
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });
  return result;
}

// normalize all input feature arrays and then dispose of the original non normalized Tensors
const FEATURE_RESULTS = normalize(INPUTS_TENSOR);
console.log("Normalized Values:");
FEATURE_RESULTS.NORMALIZED_VALUES.print();

console.log("Min Values:");
FEATURE_RESULTS.MIN_VALUES.print();

console.log("Max Values:");
FEATURE_RESULTS.MAX_VALUES.print();

//note: any Tensors not part of tf.tidy need to be cleaned up here at the end as it will now train on the normalized values as a new tensor returned from the normalized function.
INPUTS_TENSOR.dispose();

// now actually create and define model architecture. This means the outputs of the first layer become the inputs for the second. Can combine many of them together.
const model = tf.sequential();

// we will use one dense layer with 1 neuron (units) and input of 2 input feature values (representing house size and number of rooms). This means each layer is fully connected to all of the inputs.
model.add(tf.layers.dense({ inputShape: [2], units: 1 }));

// giving a summary of whats happening
model.summary();

//calling the train function which is defined below
train();

// ! training starts here
async function train() {
  const LEARNING_RATE = 0.01; // choose learning rate that's suitable for the data we are using.

  //  compile the model with the defined learning rate and specify a loss function to use.
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  //finally do the training itself
  let results = await model.fit(
    FEATURE_RESULTS.NORMALIZED_VALUES,
    OUTPUTS_TENSOR,
    {
      validationSplit: 0.15, // take aside 15% of the data to use for validation testing.
      shuffle: true, // ensure data is shuffled in case it was in an order
      batchSize: 64, // as we have a lot of training data, batch size is set to 64. This represents the number of examples it will try before it calculates the average loss and then update the weights and bias of the model.
      epochs: 10, // go over the data 10 times
    }
  );

  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

  console.log(
    "Average error loss: " +
      Math.sqrt(results.history.loss[results.history.loss.length - 1])
  );
  console.log(
    "Average validation error loss: " +
      Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])
  );

  evaluate(); // once trained evaluate the model
}

function evaluate() {
  // predict answer for a single piece of data
  tf.tidy(function () {
    let newInput = normalize(
      tf.tensor2d([[750, 1]]),
      FEATURE_RESULTS.MIN_VALUES,
      FEATURE_RESULTS.MAX_VALUES
    );

    let output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });

  // finally when you no longer need to make any more predictions clean up remaining Tensors.
  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();

  console.log(tf.memory().numTensors);
}
