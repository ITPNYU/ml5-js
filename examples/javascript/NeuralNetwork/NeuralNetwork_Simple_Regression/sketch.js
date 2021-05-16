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
let counter = 0;
let canvas, ctx;

const options = {
  task: "regression",
  debug: true,
};

window.onload = function () {
  canvas = document.querySelector("canvas");
  ctx = canvas.getContext("2d");

  canvas.width = 400;
  canvas.height = 400;
  canvas.style.background = "#eee";

  nn = ml5.neuralNetwork(options);

  // console.log(nn);
  createTrainingData();
  nn.normalizeData();

  const trainingOptions = {
    batchSize: 24,
    epochs: 10,
  };

  nn.train(trainingOptions, finishedTraining); // if you want to change the training options
  // nn.train(finishedTraining); // use the default training options
};

function finishedTraining() {
  if (counter < 400) {
    nn.predict([counter], (err, results) => {
      if (err) {
        console.log(err);
        return;
      }
      console.log(results[0]);
      const prediction = results[0];
      const x = counter;
      const y = prediction.value;
      ctx.fillStyle = "rgb(255, 0, 0)";
      ctx.strokeStyle = "black";
      ctx.strokeRect(x + 5, y + 5, 10, 10);
      ctx.fillRect(x + 6, y + 6, 8, 8);

      counter++;
      finishedTraining();
    });
  }
}

function createTrainingData() {
  for (let i = 0; i < canvas.width; i += 10) {
    /*
     * This is equal to floor(random.(5, 20))
     * Using the formula :-
     * → (Math.random() * (max - min)) + min;
     */
    const iters = Math.floor(Math.random() * 15) + 5;
    const spread = 50;
    for (let j = 0; j < iters; j += 1) {
      const data = [
        i,
        canvas.height -
          i +
          Math.floor(Math.random() * (spread + spread)) -
          spread,
      ];
      ctx.fillStyle = "rgb(0, 0, 255)";
      ctx.beginPath();
      ctx.arc(data[0], data[1], 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.closePath();
      nn.addData([data[0]], [data[1]]);
    }
  }
}
