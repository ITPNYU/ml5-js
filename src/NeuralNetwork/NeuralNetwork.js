import * as tf from '@tensorflow/tfjs';
import callCallback from '../utils/callcallback';
import {
  saveBlob
} from '../utils/io';


class NeuralNetwork {

  constructor(_options) {
    const options = _options || {};
    // config
    this.config = options;

    this.isTrained = false;
    this.isCompiled = false;
    this.isLayered = false;

    this.model = null;

    this.init();
  }

  init(){
    this.createModel();
  }

  // eslint-disable-next-line class-methods-use-this
  createModel(_type = 'sequential') {
    switch (_type.toLowerCase()) {
      case 'sequential':
        this.model = tf.sequential();
        return this.model;
      default:
        this.model = tf.sequential();
        return this.model;
    }
  }

  /**
   * {inputShape: [1], units: 1, useBias: true}  
   * basic:
   * tf.layers.dense (input)
   * tf.layers.dense (output)
   * convolutional nn: 
   * tf.layers.conv2d, tf.layers.maxPooling2d, 
   * tf.layers.maxPooling2d, tf.layers.flatten(),
   * tf.layers.dense
   * @param {*} _layerOptions 
   */
  addLayer(_layerOptions) {
    const LAYER_OPTIONS = _layerOptions || {};
    this.model.add(LAYER_OPTIONS);

    // check if it has at least an input and output layer
    if(this.model.layers.length >=2){
      this.isLayered = true;
    }
    
  }

  /**
   * {
   *    optimizer: tf.train.adam(),
   *    loss: tf.losses.meanSquaredError,
   *    metrics: ['mse'],
   *  } 
   * @param {*} _trainingOptions 
   */
  compile(_modelOptions) {
    this.model.compile(_modelOptions);
    this.isCompiled = true;
  }

  /**
   * 
   * @param {*} learningRate 
   * @param {*} optimizer 
   */
  setOptimizerFunction(learningRate, optimizer) {
    return optimizer.call(this, learningRate);
  }

  /**
   * train(_options, _cb){ 
   * @param {*} _options 
   * @param {*} _cb 
   */
  train(_options, _cb) {
    return callCallback(this.trainInternal(_options), _cb);
  }

  /**
   * trainInternal
   * @param {*} _options 
   */
  async trainInternal(_options) {
    const TRAINING_OPTIONS = _options;

    const xs = TRAINING_OPTIONS.inputs;
    const ys = TRAINING_OPTIONS.outputs;

    const {
      batchSize,
      epochs,
      shuffle,
      validationSplit,
      whileTraining
    } = TRAINING_OPTIONS;

    await this.model.fit(xs, ys, {
      batchSize,
      epochs,
      shuffle,
      validationSplit,
      callbacks: [{
        onEpochEnd: whileTraining
      }]
    })

    xs.dispose();
    ys.dispose();

    this.isTrained = true;
  }

  /**
   * predict
   * @param {*} _inputs 
   * @param {*} _cb 
   */
  predict(_inputs, _metaOrCb = null, _cb) {
    let cb;
    let meta;
    if(typeof _metaOrCb === 'function'){
      cb = _metaOrCb;
      meta = null;
    } else if(_metaOrCb instanceof Object ){
      meta = _metaOrCb;
      cb = _cb;
    }
    return callCallback(this.predictInternal(_inputs, meta), cb);
  }

  /**
   * classify
   * @param {*} _inputs 
   * @param {*} _cb 
   */
  classify(_inputs, _metaOrCb = null, _cb) {
    let cb;
    let meta;
    if(typeof _metaOrCb === 'function'){
      cb = _metaOrCb;
      meta = null;
    } else if(_metaOrCb instanceof Object ){
      meta = _metaOrCb;
      cb = _cb;
    }
    return callCallback(this.classifyInternal(_inputs, meta), cb);
  }  

  /**
   * 
   * @param {*} _inputs 
   * @param {*} _meta 
   */
  async classifyInternal(_inputs, _meta = null) {
    const output = tf.tidy(() => {
      return this.model.predict(_inputs);
    })
    const result = await output.array();

    if (_meta !== null) {
      const label = Object.keys(_meta.outputs)[0]
      const vals = Object.entries(_meta.outputs[label].legend);

      const results = vals.map((item, idx) => {
        return {
          [item[0]]: result[0][idx],
          label: item[0],
          confidence: result[0][idx]
        };
      }).sort( (a, b) => b.confidence - a.confidence)

      output.dispose();
      _inputs.dispose();
      return results;
    }

    output.dispose();
    _inputs.dispose();

    return result;
  }

  // eslint-disable-next-line class-methods-use-this
  unNormalizeValue(value, min, max) {
    return ((value * (max - min)) + min)
  }

  // eslint-disable-next-line class-methods-use-this
  async predictInternal(_inputs, _meta) {
    const output = tf.tidy(() => {
      return this.model.predict(_inputs);
    })
    const result = await output.array();

    if (_meta !== null) {
      const labels = Object.keys(_meta.outputs);
      const results = labels.map((item, idx) => {
        
        // check to see if the data were normalized
        // if not, then send back the values, otherwise
        // unnormalize then return
        let val;
        if(_meta.isNormalized){
          const { min, max} = _meta.outputs[item];
          val = this.unNormalizeValue(result[0][idx], min, max)
        } else{
          val = result[0][idx]
        }
        
        return {
          [labels[idx]]: val,
          label: item,
          value: val
        };
      })

      output.dispose();
      _inputs.dispose();

      return results;
    }

    output.dispose();
    _inputs.dispose();

    return result;
  }

  // TODO: 
  
  //  /**
  //  * predictMultiple
  //  * @param {*} _inputs 
  //  * @param {*} _cb 
  //  */
  // predictMultiple(_inputs, _cb) {
  //   return callCallback(this.predictMultipleInternal(_inputs), _cb);
  // }

  // /**
  //  * classifyMultiple
  //  * @param {*} _inputs 
  //  * @param {*} _cb 
  //  */
  // classifyMultiple(_inputs, _cb) {
  //   this.predictMultiple(_inputs, _cb);
  // }

  // // eslint-disable-next-line class-methods-use-this
  // async predictMultipleInternal(_inputs) {
  //   const output = tf.tidy(() => {
  //     return this.model.predict(_inputs);
  //   })
  //   const result = await output.array();
  //   output.dispose();
  //   _inputs.dispose();
  //   return result;
  // }


  // eslint-disable-next-line class-methods-use-this
  async save(nameOrCb, cb) {
    let modelName;
    let callback;

    if(typeof nameOrCb === 'function'){
      modelName = 'model';
      callback = nameOrCb;
    } else if (typeof nameOrCb === 'string'){
      modelName = nameOrCb

      if(typeof cb === 'function'){
        callback = cb
      } 
    } else{
      modelName = 'model';
    }

    this.model.save(tf.io.withSaveHandler(async (data) => {
      
      this.weightsManifest = {
        modelTopology: data.modelTopology,
        weightsManifest: [{
          paths: [`./${modelName}.weights.bin`],
          weights: data.weightSpecs,
        }]
      };

      await saveBlob(data.weightData, `${modelName}.weights.bin`, 'application/octet-stream');
      await saveBlob(JSON.stringify(this.weightsManifest), `${modelName}.json`, 'text/plain');
      if (callback) {
        callback();
      }
    }));
  }

  /**
   * loads the model and weights
   * @param {*} filesOrPath 
   * @param {*} callback 
   */
  // eslint-disable-next-line class-methods-use-this
  async load(filesOrPath = null, callback){
    if (filesOrPath instanceof FileList) {
      
      const files = await Promise.all(
        Array.from(filesOrPath).map( async (file) => {
          if (file.name.includes('.json') && !file.name.includes('_meta')) {
            return {name:"model", file}
          } else if ( file.name.includes('.json') && file.name.includes('_meta.json')) {
            const modelMetadata = await file.text();
            return {name: "metadata", file:modelMetadata}
          } else if (file.name.includes('.bin')) {
            return {name:"weights", file}
          }
          return {name:null, file:null}
        })
       )

      const model = files.find(item => item.name === 'model').file;
      const weights = files.find(item => item.name === 'weights').file;

      // load the model
      this.model = await tf.loadLayersModel(tf.io.browserFiles([model, weights]));

    } else if(filesOrPath instanceof Object){
      // filesOrPath = {model: URL, metadata: URL, weights: URL}

      let modelJson = await fetch(filesOrPath.model);
      modelJson = await modelJson.text();
      const modelJsonFile = new File([modelJson], 'model.json', {type: 'application/json'});

      let weightsBlob = await fetch(filesOrPath.weights);
      weightsBlob = await weightsBlob.blob();
      const weightsBlobFile = new File([weightsBlob], 'model.weights.bin', {type: 'application/macbinary'});

      this.model = await tf.loadLayersModel(tf.io.browserFiles([modelJsonFile, weightsBlobFile]));

    } else {
      this.model = await tf.loadLayersModel(filesOrPath);
    }

    this.isCompiled = true;
    this.isLayered = true;
    this.isTrained = true;

    if (callback) {
      callback();
    }
    return this.model;
  }


}
export default NeuralNetwork;