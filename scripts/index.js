/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from '../imagenet_classes';
import { renderEvaluateTable } from './ui';

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;

let mobilenet;
const mobilenetDemo = async () => {
  status('Loading model...');

  mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Make a prediction through the locally hosted cat.jpg.
  // const catElement = document.getElementById('cat');
  // if (catElement.complete && catElement.naturalHeight !== 0) {
  //   predict(catElement);
  //   catElement.style.display = '';
  // } else {
  //   catElement.onload = () => {
  //     predict(catElement);
  //     catElement.style.display = '';
  //   }
  // }

  document.getElementById('file-container').style.display = '';
};
async function callpredic(file){
  
  // var imageBuffer = request.file.buffer;
  // var imageName = 'public/images/map.png';
  // fs.createWriteStream(imageName).write(imageBuffer);
  // var data = new Uint8Array([
    //   137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,8,0,0,
    //   0,8,8,2,0,0,0,75,109,41,220,0,0,0,34,73,68,65,84,8,215,99,120,
    //   173,168,135,21,49,0,241,255,15,90,104,8,33,129,83,7,97,163,136,
    //   214,129,93,2,43,2,0,181,31,90,179,225,252,176,37,0,0,0,0,73,69,
    //   78,68,174,66,96,130]);
    // var blob = new Blob([data], { type: "image/png" });
    // var url = URL.createObjectURL(blob);
    // var img = new Image();
    // img.src = url;
    // console.log("data length: " + data.length);
    // console.log("url: " + url);
    // document.body.appendChild(img);
    
  console.log('data lenght' + file.name)
  let formData  = new FormData();
  formData.append('file', file);
  // formData.append('name', file.name);
  console.log(file.name)
  // fileofBlob = new Blob([imgdata],{type: 'image/png'})
  // var url = URL.createObjectURL(fileofBlob);
  // console.log(url)
  // var imgel = new Image();
  // imgel.src = url;
  // predict(imgel)
  // formData.append('img', fileofBlob)

  let xhr = new XMLHttpRequest();
  xhr.open('POST', '/predict');
  xhr.onload = () => {
    if (xhr.status === 200){
      // console.log("Something went wrong, Name is now " + xhr.responseText);
      var jsonobj = JSON.parse(xhr.responseText);
      var className = jsonobj.class_name
      var classid = jsonobj.class_id
      var probability = jsonobj.prob
      renderEvaluateTable([file.name,classid],[className],probability);
      const ele = document.getElementById(className);
      ele.style.display = ''
    }else if(xhr.status!==200){
      console.log("Rquest Failed, Returned status of " + xhr.status) 
      
    }
  };
  console.log(formData)
  xhr.send(formData)
}
/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  status('Predicting...');

  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return mobilenet.predict(batched);
  });

  const fileName = imgElement.uid;
  await showImags(imgElement);
  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
  `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
  
  // Show the classes in the DOM.
  console.log('entering render')
  console.log(classes[0].probability,classes[0].className)
  renderEvaluateTable([fileName,fileName],[classes[0].className],classes[0].probability);
  // showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(logits, topK) {
  const values = await logits.data();
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}
export async function showImags(imgElement){
  const licontainter = document.createElement('li')
  licontainter.id = "sample";
  licontainter.appendChild(imgElement)
  const samples = document.querySelector('#samples');
  samples.appendChild(licontainter)
  // predict(imgElement)
}

//
// UI
//
// rewrite the showresults, and show the image search result in matrix or inside the container
function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}


const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    // reader.onload = e => {
    //   // Fill the image & call predict.
    //   let img = document.createElement('img');
    //   img.src = e.target.result;
    //   img.id = i;
    //   img.uid = f.name;
    //   console.log(f.name);
    //   img.width = IMAGE_SIZE;
    //   img.height = IMAGE_SIZE;
    //   img.onload = () => predict(img);
    // };
    reader.onload = e => {
      // console.log(e.target.result)
      // callpredic(e.target.result)
      callpredic(f)
    };
    // reader.onload = e => {
    //   console.log(e.target.result)
    //   callpredic(e.target.result)
    // }

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
    // reader.readAsArrayBuffer(f)
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');
export function setManualInputWinnerMessage(message) {
  const winnerElement = document.getElementById('winner');
  winnerElement.textContent = message;
}
mobilenetDemo();
