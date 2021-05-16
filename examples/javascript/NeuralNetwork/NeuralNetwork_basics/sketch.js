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

window.onload = function () {
  // Options for Neural Network
  const options = {
    task: "regression",
    debug: true,
  };
  // Create Neural Network
  nn = ml5.neuralNetwork(options);

  trainModel();

  // Train the model
  const trainBtn = document.createElement("button");
  trainBtn.textContent = "Train Model";
  trainBtn.addEventListener("click", function () {
    trainModel();
  });

  document.body.appendChild(trainBtn);

  // Predict
  const predictBtn = document.createElement("button");
  predictBtn.textContent = "Predict";
  predictBtn.addEventListener("click", function () {
    predict();
  });

  document.body.appendChild(predictBtn);

  // Save and download the model
  const saveBtn = document.createElement("button");
  saveBtn.textContent = "Save Model";
  saveBtn.addEventListener("click", function () {
    nn.save();
  });

  document.body.appendChild(saveBtn);

  // Load the model from local files
  const loadLocalBtn = document.createElement("button");
  loadLocalBtn.textContent = "Load the model from local files";
  loadLocalBtn.addEventListener("click", function () {
    nn.load("model/model.json", function () {
      console.log("Model Loaded!");
    });
  });

  document.body.appendChild(loadLocalBtn);

  // Load model
  const loadBtn = document.querySelector("#load");
  loadBtn.addEventListener("change", function () {
    // console.log(this.files);

    // Basic file handling
    const modelDetails = {
      model: "",
      metadata: "",
      weights: "",
    };

    for (let i = 0; i < this.files.lenght; i++) {
      let file = this.files[i];
      if (file.name.endsWith(".json") && modelDetails.model !== "") {
        modelDetails.model = file;
      } else if (
        file.name.endsWith(".weights.bin") &&
        modelDetails.weights !== ""
      ) {
        modelDetails.weights = file;
      } else if (
        file.name.endsWith("_meta.bin") &&
        modelDetails.metadata !== ""
      ) {
        modelDetails.metadata = file;
      }
    }

    nn.load(modelDetails, function () {
      console.log("Model Loaded!");
    });
  });
};

function trainModel() {
  // Add training data
  // const trainingInput = [-0.6, 1, 0.25];
  // const trainingTarget = [0.3, 0.9];
  let a, b, c;
  let trainingTarget;
  for (let i = 0; i < 500; i += 1) {
    if (i % 2) {
      /* Modifiying Math.random to have a min and max value
       * Using formula => (Math.random() * (max - min)) + min
       */
      a = Math.random() * (0.16 - 0) + 0;
      b = Math.random() * (0.32 - 0.16) + 0.16;
      c = Math.random() * (0.5 - 0.32) + 0.32;
      trainingTarget = [0, 0];
    } else {
      a = Math.random() * (0.66 - 0.5) + 0.5;
      b = Math.random() * (0.82 - 0.66) + 0.66;
      c = Math.random() * (1 - 0.82) + 0.82;
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
