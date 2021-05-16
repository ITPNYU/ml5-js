// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
Image classification using MobileNet and p5.js
This example uses a callback pattern to create the classifier
=== */
let nn;

function setup() {
  // Options for Neural Network
  const options = {
    task: "regression",
    debug: true,
  };
  // Create Neural Network
  nn = ml5.neuralNetwork(options);

  trainModel();

  // Train the model
  const trainBtn = createButton("Train Model");
  trainBtn.position(10, 50);
  trainBtn.mousePressed(function() {
    trainModel();
  });

  // Predict
  const predictBtn = createButton("Predict");
  predictBtn.position(10, 70);
  predictBtn.mousePressed(function() {
    predict();
  });

  // Save and download the model
  const saveBtn = createButton("Save Model");
  saveBtn.position(10, 90);
  saveBtn.mousePressed(function() {
    nn.save();
  });

  // Load the model from local files
  const loadLocalBtn = createButton("Load the model from local files");
  loadLocalBtn.position(10, 110);
  loadLocalBtn.mousePressed(function() {
    nn.load("model/model.json", function() {
      console.log("Model Loaded!");
    });
  });

  // Load model
  const loadBtn = select("#load");
  loadBtn.changed(function() {
    nn.load(loadBtn.elt.files, function() {
      console.log("Model Loaded!");
    });
  });
}

function trainModel() {
  // Add training data
  // const trainingInput = [-0.6, 1, 0.25];
  // const trainingTarget = [0.3, 0.9];
  let a, b, c;
  let trainingTarget;
  for (let i = 0; i < 500; i += 1) {
    if (i % 2) {
      a = random(0, 0.16);
      b = random(0.16, 0.32);
      c = random(0.32, 0.5);
      trainingTarget = [0, 0];
    } else {
      a = random(0.5, 0.66);
      b = random(0.66, 0.82);
      c = random(0.82, 1);
      trainingTarget = [1, 1];
    }

    const trainingInput = [a, b, c];
    // const trainingTarget = [0, 1];

    // nn.data.addData(
    //   {
    //     input0: trainingInput[0],
    //     input1: trainingInput[1],
    //     input2: trainingInput[2],
    //   },
    //   {
    //     output0: trainingTarget[0],
    //     output1: trainingTarget[1],
    //   });

    nn.addData(trainingInput, trainingTarget);
  }

  const trainingOptions = {
    epochs: 32,
    batchSize: 12,
  };
  // Train
  nn.normalizeData();
  nn.train(trainingOptions, finishedTraining);
}

// Training callback
function finishedTraining() {
  predict();
}

function predict() {
  const a = 0.1;
  const b = 0.2;
  const c = 0.4;
  const input = [a, b, c];
  // we should expect [0,0]
  nn.predict(input, gotResults);
}

function gotResults(error, results) {
  if (error) console.log(error);
  if (results) {
    console.log(results);
  }
}
