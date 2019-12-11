import * as tf from '@tensorflow/tfjs';
import NeuralNetwork from './NeuralNetwork';
import NeuralNetworkData from './NeuralNetworkData';
import NeuralNetworkVis from './NeuralNetworkVis';
import callCallback from '../utils/callcallback';

import nnUtils from './NeuralNetworkUtils';

const DEFAULTS = {
  inputs: [],
  outputs: [],
  dataUrl: null,
  modelUrl: null,
  layers: [],
  task: null,
  debug: false,
  learningRate: 0.2,
  hiddenUnits: 16
}
class DiyNeuralNetwork {

  constructor(options, cb) {
    this.callback = cb;
    this.options = {
      ...DEFAULTS,
      ...options
    } || DEFAULTS;

    this.neuralNetwork = new NeuralNetwork();
    this.neuralNetworkData = new NeuralNetworkData();
    this.neuralNetworkVis = new NeuralNetworkVis();

    this.data = {
      training: []
    }

    this.ready = false;
    this.init(this.callback);

  }

  /** 
   * ////////////////////////////////////////////////////////////
   * Initialization
   * ////////////////////////////////////////////////////////////
   */

  /**
   * init
   * @param {*} callback 
   */
  init(callback) {
    if (this.options.dataUrl !== null) {
      this.ready = this.loadDataFromUrl(this.options, callback);
    } else if (this.options.modelUrl !== null) {
      // will take a URL to model.json, an object, or files array
      this.ready = this.load(this.options.modelUrl, callback);
    } else {
      this.ready = true;
    }
  }


  /** 
   * ////////////////////////////////////////////////////////////
   * Data Handling
   * ////////////////////////////////////////////////////////////
   */

  /**
   * loadData
   * @param {*} options 
   * @param {*} callback 
   */
  loadDataFromUrl(options, callback) {
    return callCallback(this.loadDataInternal(options), callback)
  }

  /**
   * loadDataInternal
   * @param {*} options 
   */
  async loadDataInternal(options) {
    const {
      dataUrl,
      inputs,
      outputs
    } = options;

    const data = await this.neuralNetworkData.loadDataFromUrl(dataUrl, inputs, outputs);

    // once the data are loaded, create the metadata 
    // and prep the data for training
    // this.createMetaDataFromData();
    this.neuralNetworkData.createMetadata(data);
    this.prepareForTraining(data);
  }

  /**
   * Prepare data for training by applying oneHot to raw
   * @param {*} dataRaw 
   */
  prepareForTraining(_dataRaw = null) {
    const dataRaw = _dataRaw === null ? this.neuralNetworkData.data.raw : _dataRaw;
    const unnormalizedTrainingData = this.neuralNetworkData.applyOneHotEncodingsToDataRaw(dataRaw);
    this.data.training = unnormalizedTrainingData;
    this.neuralNetworkData.isWarmedUp = true;

    return unnormalizedTrainingData;
  }

  /**
   * normalizeData
   * @param {*} _dataRaw 
   * @param {*} _meta 
   */
  normalizeData(_dataRaw = null) {

    const dataRaw = _dataRaw === null ? this.neuralNetworkData.data.raw : _dataRaw;

    if (!this.neuralNetworkData.isMetadataReady) {
      this.neuralNetworkData.createMetadata(dataRaw);
    }

    if (!this.neuralNetworkData.isWarmedUp) {
      this.prepareForTraining(dataRaw);
    }

    const trainingData = this.neuralNetworkData.normalizeDataRaw(dataRaw)

    // set this equal to the training data
    this.data.training = trainingData;

    // set isNormalized to true
    this.neuralNetworkData.meta.isNormalized = true;

    return trainingData;
  }

  /**
   * addData
   * @param {Array | Object} xInputs
   * @param {Array | Object} yInputs 
   * @param {*} options 
   */
  addData(xInputs, yInputs, options = null) {

    const {
      inputs,
      outputs
    } = this.options;

    // get the input and output labels
    // or infer them from the data
    let inputLabels;
    let outputLabels;

    if (options !== null) {
      // eslint-disable-next-line prefer-destructuring
      inputLabels = options.inputLabels;
      // eslint-disable-next-line prefer-destructuring
      outputLabels = options.outputLabels;
    } else if (
      (inputs.length > 0) && (outputs.length > 0)) {
      // if the inputs and outputs labels have been defined
      // in the constructor
      if (inputs.every(item => item instanceof String)) {
        inputLabels = inputs;
      }
      if (outputs.every(item => item instanceof String)) {
        outputLabels = outputs;
      }
    } else if (typeof xInputs === "object" && typeof yInputs === "object") {
      inputLabels = Object.keys(xInputs);
      outputLabels = Object.keys(yInputs);
    } else {
      inputLabels = nnUtils.createLabelsFromArrayValues(xInputs, 'input')
      outputLabels = nnUtils.createLabelsFromArrayValues(yInputs, 'output')
    }

    // Make sure that the inputLabels and outputLabels are arrays
    if (!(inputLabels instanceof Array)) {
      throw new Error('inputLabels must be an array')
    }
    if (!(outputLabels instanceof Array)) {
      throw new Error('outputLabels must be an array')
    }

    const xs = nnUtils.formatDataAsObject(xInputs, inputLabels);
    const ys = nnUtils.formatDataAsObject(yInputs, outputLabels);

    this.neuralNetworkData.addData(xs, ys);
  }



  /** 
   * ////////////////////////////////////////////////////////////
   * Model handling and training data preparation
   * ////////////////////////////////////////////////////////////
   */

  /**
   * convertTrainingDataToTensors
   * @param {*} _trainingData 
   * @param {*} _meta 
   */
  convertTrainingDataToTensors(_trainingData = null, _meta = null) {
    const trainingData = _trainingData === null ? this.data.training : _trainingData;
    const meta = _meta === null ? this.neuralNetworkData.meta : _meta;

    return this.neuralNetworkData.convertRawToTensors(trainingData, meta);
  }

  /**
   * train
   * @param {*} optionsOrCallback 
   * @param {*} optionsOrWhileTraining 
   * @param {*} callback 
   */
  train(optionsOrCallback, optionsOrWhileTraining, callback) {
    let options;
    let whileTrainingCb;
    let finishedTrainingCb;
    if (typeof optionsOrCallback === 'object' &&
      typeof optionsOrWhileTraining === 'function' &&
      typeof callback === 'function'
    ) {
      options = optionsOrCallback;
      whileTrainingCb = optionsOrWhileTraining;
      finishedTrainingCb = callback;
    } else if (typeof optionsOrCallback === 'object' &&
      typeof optionsOrWhileTraining === 'function') {
      options = optionsOrCallback;
      whileTrainingCb = null;
      finishedTrainingCb = optionsOrWhileTraining;
    } else if (typeof optionsOrCallback === 'function' &&
      typeof optionsOrWhileTraining === 'function'
    ) {
      options = {};
      whileTrainingCb = optionsOrCallback;
      finishedTrainingCb = optionsOrWhileTraining;
    } else {
      options = {};
      whileTrainingCb = null;
      finishedTrainingCb = optionsOrCallback;
    }

    this.trainInternal(options, whileTrainingCb, finishedTrainingCb);
  }

    /**
     * train
     * @param {*} _options 
     * @param {*} _cb 
     */
    trainInternal(_options, whileTrainingCb, finishedTrainingCb) {
      
      const options = {
        epochs: 10,
        batchSize: 32,
        validationSplit: 0.1,
        whileTraining: null,
        ..._options
      };

      // if debug mode is true, then use tf vis
      if (this.options.debug === true) {
        options.whileTraining = [
          this.neuralNetworkVis.trainingVis(),
          {
            onEpochEnd: null
          }
        ]
      } else {
        // if not use the default training
        // options.whileTraining = whileTrainingCb === null ? [{
        //     onEpochEnd: (epoch, loss) => {
        //       console.log(epoch, loss.loss)
        //     }
        //   }] :
        //   [{
        //     onEpochEnd: whileTrainingCb
        //   }];
        options.whileTraining = [{
          onEpochEnd: whileTrainingCb
        }];
      }


      // if metadata needs to be generated about the data
      if (!this.neuralNetworkData.isMetadataReady) {
        // if the inputs are defined as an array of [img_width, img_height, channels]
        const {inputs} = this.options;
        const inputShape = inputs.every(item => typeof item === 'number') && inputs.length > 0 ? inputs : null;
        this.neuralNetworkData.createMetadata(this.neuralNetworkData.data.raw, inputShape);
      }

      // if the data still need to be summarized, onehotencoded, etc
      if (!this.neuralNetworkData.isWarmedUp) {
        this.prepareForTraining(this.neuralNetworkData.data.raw);
      }

      // if inputs and outputs are not specified
      // in the options, then create the tensors
      // from the this.neuralNetworkData.data.raws
      if (!options.inputs && !options.outputs) {
        const {
          inputs,
          outputs
        } = this.convertTrainingDataToTensors();
        options.inputs = inputs;
        options.outputs = outputs;
      }

      // check to see if layers are passed into the constructor
      // then use those to create your architecture
      if (!this.neuralNetwork.isLayered) {
        this.options.layers = this.createNetworkLayers(this.options.layers, this.neuralNetworkData.meta);
      }

      // if the model does not have any layers defined yet
      // then use the default structure
      if (!this.neuralNetwork.isLayered) {
        this.options.layers = this.addDefaultLayers(this.options.task, this.neuralNetworkData.meta);
      }

      if (!this.neuralNetwork.isCompiled) {
        // compile the model with defaults
        this.compile()
      }

      // train once the model is compiled
      this.neuralNetwork.train(options, finishedTrainingCb);
    }

  /** 
   * ////////////////////////////////////////////////////////////
   * Layer handling
   * ////////////////////////////////////////////////////////////
   */

  /**
   * addLayer
   * @param {*} options 
   */
  addLayer(options) {
    this.neuralNetwork.addLayer(options);
  }


  /**
   * add custom layers in options
   */
  createNetworkLayers(layerJsonArray, meta) {
    const layers = [...layerJsonArray];

    const {
      inputUnits,
      outputUnits
    } = Object.assign({}, meta);
    const layersLength = layers.length;

    if (!(layers.length >= 2)) {
      return false;
    }

    // set the inputShape
    layers[0].inputShape = layers[0].inputShape ? layers[0].inputShape : inputUnits;
    // set the output units
    const lastIndex = layersLength - 1;
    const lastLayer = layers[lastIndex];
    lastLayer.units = lastLayer.units ? lastLayer.units : outputUnits;

    layers.forEach(layer => {
      this.addLayer(tf.layers[layer.type](layer));
    })

    return layers;

  }

  // /**
  //  * createDenseLayer
  //  * @param {*} _options 
  //  */
  // // eslint-disable-next-line class-methods-use-this
  // createDenseLayer(_options) {
  //   const options = Object.assign({}, {
  //     units: this.options.hiddenUnits,
  //     activation: 'relu',
  //     ..._options
  //   });
  //   return tf.layers.dense(options);
  // }

  // /**
  //  * createConv2dLayer
  //  * @param {*} _options 
  //  */
  // // eslint-disable-next-line class-methods-use-this
  // createConv2dLayer(_options) {
  //   const options = Object.assign({}, {
  //     kernelSize: 5,
  //     filters: 8,
  //     strides: 1,
  //     activation: 'relu',
  //     kernelInitializer: 'varianceScaling',
  //     ..._options
  //   })

  //   return tf.layers.conv2d(options);
  // }

  /**
   * addDefaultLayers
   * @param {*} _task 
   */
  addDefaultLayers(task, meta) {
    let layers;
    switch (task.toLowerCase()) {
      // if the task is classification
      case 'classification':

        layers = [{
            type: 'dense',
            units: this.options.hiddenUnits,
            activation: 'relu',
          },
          {
            type: 'dense',
            activation: 'softmax'
          }
        ]

        return this.createNetworkLayers(layers, meta)
        // if the task is regression
      case 'regression':

        layers = [{
            type: 'dense',
            units: this.options.hiddenUnits,
            activation: 'relu',
          },
          {
            type: 'dense',
            activation: 'sigmoid'
          }
        ]
        return this.createNetworkLayers(layers, meta);
        // if the task is imageClassification
      case 'imageclassification':

        layers = [{
            type: 'conv2d',
            filters: 2,
            kernelSize: 2,
            strides: 2,
            activation: 'relu',
            kernelInitializer: 'varianceScaling',
          },
          {
            type: 'maxPooling2d',
            poolSize: [1, 1],
            strides: [1, 1]
          },
          {
            type: 'conv2d',
            filters: 1,
            kernelSize: 1,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
          },
          {
            type: 'maxPooling2d',
            poolSize: [1, 1],
            strides: [1, 1]
          },
          {
            type: 'flatten'
          },
          {
            type: 'dense',
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
          }
        ]
        return this.createNetworkLayers(layers, meta);

      default:
        console.log('no imputUnits or outputUnits defined')
        layers = [{
            type: 'dense',
            units:this.options.hiddenUnits,
            activation: 'relu',
          },
          {
            type: 'dense',
            activation: 'sigmoid'
          }
        ]
        return this.createNetworkLayers(layers, meta);
    }
  }

  /**
   * compile the model
   * @param {*} _options 
   */
  compile(_modelOptions = null, _learningRate = null) {
    const LEARNING_RATE = _learningRate === null ? this.options.learningRate : _learningRate;

    let options = {};

    if (_modelOptions !== null) {
      options = {
        ..._modelOptions
      }
    } else if ((this.options.task === 'classification') || (this.options.task === 'imageClassification')) {
      options = {
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.sgd,
        metrics: ['accuracy'],
      }
    } else if (this.options.task === 'regression') {
      options = {
        loss: 'meanSquaredError',
        optimizer: tf.train.adam,
        metrics: ['accuracy'],
      }
    }

    options.optimizer = options.optimizer ?
      this.neuralNetwork.setOptimizerFunction(LEARNING_RATE, options.optimizer) :
      this.neuralNetwork.setOptimizerFunction(LEARNING_RATE, tf.train.sgd)

    this.neuralNetwork.compile(options);

    // if debug mode is true, then show the model summary
    if (this.options.debug) {
      this.neuralNetworkVis.modelSummary({
        name: 'Model Summary'
      }, this.neuralNetwork.model);
    }
  }

  //   /** 
  //    * ***********************************************
  //    * Input handling for prediction / classification
  //    * ***********************************************
  //   */

  //   /**
  //    * format the inputs for prediction
  //    * this means applying onehot or normalization
  //    * so that the user can use original data units rather 
  //    * than having to normalize
  //    * @param {*} _input 
  //    * @param {*} meta 
  //    * @param {*} inputHeaders 
  //    */
  //   formatInputsForPrediction(_input, meta, inputHeaders) {
  //     let inputData = [];

  //     // TODO: check to see if it is a nested array 
  //     // to run predict or classify on a batch of data

  //     if (_input instanceof Array) {
  //       inputData = inputHeaders.map((prop, idx) => {
  //         return this.isOneHotEncodedOrNormalized(_input[idx], prop, meta.inputs);
  //       });

  //     } else if (_input instanceof Object) {
  //       // TODO: make sure that the input order is preserved!
  //       inputData = inputHeaders.map(prop => {
  //         return this.isOneHotEncodedOrNormalized(_input[prop], prop, meta.inputs);
  //       });
  //     }

  //     // inputData = tf.tensor([inputData.flat()])
  //     inputData = inputData.flat()

  //     return inputData;
  //   }

  //   /**
  //    * formatInputsForPredictionAll
  //    * @param {*} _input 
  //    * @param {*} meta 
  //    * @param {*} inputHeaders 
  //    */
  //   formatInputsForPredictionAll(_input, meta, inputHeaders) {
  //     let output;

  //     if (_input instanceof Array) {
  //       if (_input.every(item => Array.isArray(item))) {

  //         output = _input.map(item => {
  //           return this.formatInputsForPrediction(item, meta, inputHeaders)
  //         })

  //         return tf.tensor(output, [_input.length, inputHeaders.length]);

  //       }
  //       output = this.formatInputsForPrediction(_input, meta, inputHeaders)
  //       return tf.tensor([output]);
  //     }

  //     output = this.formatInputsForPrediction(_input, meta, inputHeaders)
  //     return tf.tensor([output]);
  //   }

  //   /**
  //    * check if the input needs to be onehot encoded or 
  //    * normalized
  //    * @param {*} _input 
  //    * @param {*} _meta 
  //    */
  //   // eslint-disable-next-line class-methods-use-this
  //   isOneHotEncodedOrNormalized(_input, _key, _meta) {
  //     const input = _input;
  //     const key = _key;

  //     let output;
  //     if (typeof _input !== 'number') {
  //       output = _meta[key].legend[input];
  //     } else {
  //       output = _input;
  //       if (this.neuralNetworkData.meta.isNormalized) {
  //         output = this.normalizeInput(_input, key, _meta);
  //       }
  //     }
  //     return output;
  //   }

  //   /**
  //    * normalize the input value
  //    * @param {*} value 
  //    * @param {*} _key 
  //    * @param {*} _meta 
  //    */
  //   normalizeInput(value, _key, _meta) {
  //     const key = _key;
  //     const {
  //       min,
  //       max
  //     } = _meta[key];
  //     return this.neuralNetworkData.normalizeValue(value, min, max);
  //   }

  //   /** 
  //    * *************************************
  //    * Prediction / Classification
  //    * *************************************
  //   */

  //   /**
  //    * predict
  //    * @param {*} _input 
  //    * @param {*} _cb 
  //    */
  //   predict(_input, _cb) {
  //     return callCallback(this.predictInternal(_input), _cb)
  //   }

  //   /**
  //    * predictMultiple
  //    * @param {*} _input 
  //    * @param {*} _cb 
  //    */
  //   predictMultiple(_input, _cb) {
  //     return callCallback(this.predictInternal(_input), _cb)
  //   }

  //   /**
  //    * classify
  //    * @param {*} _input 
  //    * @param {*} _cb 
  //    */
  //   classify(_input, _cb) {
  //     return callCallback(this.classifyInternal(_input), _cb)
  //   }

  //   /**
  //    * classifyMultiple
  //    * @param {*} _input 
  //    * @param {*} _cb 
  //    */
  //   classifyMultiple(_input, _cb) {
  //     return callCallback(this.classifyInternal(_input), _cb)
  //   }

  //   /**
  //    * predict
  //    * @param {*} _input 
  //    * @param {*} _cb 
  //    */
  //   async predictInternal(_input) {

  //     const {
  //       meta
  //     } = this.neuralNetworkData;
  //     const headers = Object.keys(meta.inputs);

  //     const inputData = this.formatInputsForPredictionAll(_input, meta, headers);

  //     const unformattedResults = await this.neuralNetwork.predict(inputData);
  //     inputData.dispose();

  //     if (meta !== null) {
  //       const labels = Object.keys(meta.outputs);

  //       const formattedResults = unformattedResults.map(unformattedResult => {
  //         return labels.map((item, idx) => {
  //           // check to see if the data were normalized
  //           // if not, then send back the values, otherwise
  //           // unnormalize then return
  //           let val;
  //           let unNormalized;
  //           if (meta.isNormalized) {
  //             const {
  //               min,
  //               max
  //             } = meta.outputs[item];
  //             val = this.neuralNetworkData.unNormalizeValue(unformattedResult[idx], min, max)
  //             unNormalized = unformattedResult[idx]
  //           } else {
  //             val = unformattedResult[idx]
  //           }

  //           const d = {
  //             [labels[idx]]: val,
  //             label: item,
  //             value: val,
  //           };

  //           // if unNormalized is not undefined, then
  //           // add that to the output 
  //           if (unNormalized) {
  //             d.unNormalizedValue = unNormalized;
  //           }

  //           return d;
  //         })
  //       })

  //       // return single array if the length is less than 2, 
  //       // otherwise return array of arrays
  //       if (formattedResults.length < 2) {
  //         return formattedResults[0];
  //       }
  //       return formattedResults;
  //     }

  //     // if no meta exists, then return unformatted results;
  //     return unformattedResults;

  //   }

  //   /**
  //    * classify
  //    * @param {*} _input 
  //    * @param {*} _cb 
  //    */
  //   async classifyInternal(_input) {
  //     const {
  //       meta
  //     } = this.neuralNetworkData;
  //     const headers = Object.keys(meta.inputs);

  //     const inputData = this.formatInputsForPredictionAll(_input, meta, headers);
  //     inputData.print()

  //     const unformattedResults = await this.neuralNetwork.classify(inputData);
  //     inputData.dispose();

  //     if (meta !== null) {
  //       const label = Object.keys(meta.outputs)[0]
  //       const vals = Object.entries(meta.outputs[label].legend);

  //       const formattedResults = unformattedResults.map(unformattedResult => {
  //         return vals.map((item, idx) => {
  //           return {
  //             [item[0]]: unformattedResult[idx],
  //             label: item[0],
  //             confidence: unformattedResult[idx]
  //           };
  //         }).sort((a, b) => b.confidence - a.confidence)
  //       });

  //       // return single array if the length is less than 2, 
  //       // otherwise return array of arrays
  //       if (formattedResults.length < 2) {
  //         return formattedResults[0];
  //       }
  //       return formattedResults;
  //     }

  //     return unformattedResults;
  //   }


  /** 
   * ////////////////////////////////////////////////////////////
   * Data Saving/loading
   * ////////////////////////////////////////////////////////////
   */

  /**
   * save data
   * @param {*} name 
   */
  saveData(name) {
    this.neuralNetworkData.saveData(name);
  }

  /**
   * load data
   * @param {*} filesOrPath 
   * @param {*} callback 
   */
  async loadData(filesOrPath = null, callback) {
    this.neuralNetworkData.loadData(filesOrPath, callback);
  }

  /** 
   * ////////////////////////////////////////////////////////////
   * Model Saving/loading
   * ////////////////////////////////////////////////////////////
   */

  /**
   * saves the model, weights, and metadata
   * @param {*} nameOrCb 
   * @param {*} cb 
   */
  save(nameOrCb, cb) {
    let modelName;
    let callback;

    if (typeof nameOrCb === 'function') {
      modelName = 'model';
      callback = nameOrCb;
    } else if (typeof nameOrCb === 'string') {
      modelName = nameOrCb

      if (typeof cb === 'function') {
        callback = cb
      }

    } else {
      modelName = 'model';
    }

    // save the model
    this.neuralNetwork.save(modelName, () => {
      this.neuralNetworkData.saveMeta(modelName, callback);
    });

  }

  /**
   * load a model and metadata
   * @param {*} filesOrPath 
   * @param {*} callback 
   */
  async load(filesOrPath = null, cb) {
    let callback;
    if (cb) {
      callback = cb
    }

    this.neuralNetwork.load(filesOrPath, () => {
      this.neuralNetworkData.loadMeta(filesOrPath, callback);

      return this.neuralNetwork.model;
    })

  }


}


const neuralNetwork = (inputsOrOptions, outputsOrCallback, callback) => {
  let options;
  let cb;

  if (inputsOrOptions instanceof Object) {
    options = inputsOrOptions;
    cb = outputsOrCallback;
  } else {
    options = {
      inputs: inputsOrOptions,
      outputs: outputsOrCallback,
    };
    cb = callback;
  }

  const instance = new DiyNeuralNetwork(options, cb);
  return instance;
}

export default neuralNetwork