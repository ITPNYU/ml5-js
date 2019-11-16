// Copyright (c) 2019 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
    COCO-SSD Object detection
    Wraps the coco-ssd model in tfjs to be used in ml5
*/

import * as cocoSsd from '@tensorflow-models/coco-ssd';
import callCallback from '../../utils/callcallback';

class CocoSsd {
    /**
     * Create CocoSsd model. Works on video and images. 
     * @param {function} constructorCallback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise
     *    that will be resolved once the model has loaded.
     */
    constructor(constructorCallback) {
        this.constructorCallback = constructorCallback;
        this.ready = callCallback(this.loadModel(), constructorCallback);
    }

    async loadModel() {
        await cocoSsd.load().then(_cocoSsdModel => {
            this.cocoSsdModel = _cocoSsdModel;
        });
    }

    /**
    * Detect objects that are in video, returns bounding box, label, and confidence scores
    * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement|ImageData} subject - Subject of the detection.
    * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise
    *    that will be resolved once the prediction is done.
    */
    async detect(subject, callback) {
        await this.ready;
        return this.cocoSsdModel.detect(subject).then((predictions) => {
            const formattedPredictions = [];
            for (let i = 0; i < predictions.length; i += 1) {
                const prediction = predictions[i];
                formattedPredictions.push({
                    label: prediction.class,
                    confidence: prediction.score,
                    x: prediction.bbox[0] / subject.width,
                    y: prediction.bbox[1] / subject.height,
                    w: prediction.bbox[2] / subject.width,
                    h: prediction.bbox[3] / subject.height,
                });
            }
            return callCallback(new Promise((resolve) => {
                resolve(formattedPredictions);
            }), callback);
        })
    }
}

export default CocoSsd;