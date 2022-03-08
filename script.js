/**
 * Normalize tensor
 * @returns Tensor
 */
function normalize(tensor, prevMin, prevMin) {
  const min = prevMin || tensor.min(),
      max = prevMin || tensor.max(),
      normalisedTensor = tensor.sub(min).div(max.sub(min));
  return normalisedTensor;
}

/**
 * Denormalize tensor
 * @returns Tensor
 */
function denormalize(tensor, min, max) {
    return tensor.mul(max.sub(min)).add(min);
}

/**
 * creates a linear regression model
 * @returns model
 */
function createModel() {
    const model = tf.sequential();
    // Our model will have no activation functions
    model.add(tf.layers.dense({
        units:1,
        inputDim:1,
        activation: 'linear',
        useBias: true
    }));

    // sgd -> gradient descend
    let optimizer = tf.train.sgd(0.1);
    model.compile({
        loss: 'meanSquaredError',
        optimizer
    })
    return model;
}

/**
 * Saves the model in local storage
 */
async function save(model) {
    await model.save(`localstorage://stockPredict`);
}

/**
 * Loads trained model if present in local storage
 */
async function load() {
    const models = await tf.io.listModels();
    if(models['localstorage://stockPredict']) {
        const model = await tf.loadLayersModel('localstorage://stockPredict');
        return model
    }
    return Promise.resolve(null);
}


function plot(points, predictedPoints) {
    const data = { values: [points, ...(predictedPoints ? [predictedPoints] : [])],
        series: ['original', ...(predictedPoints ? ['prediction'] : [])] };

    const surface = { name: 'ICICI Bank stock price prediction' };
    tfvis.render.scatterplot(surface, data, {xLabel: 'Open', yLabel: 'Close'});            
}

async function loadAndTrainModel() {
    let dataset = tf.data.csv('http://localhost:4000/ICICIBANK.csv');
    let points = dataset.map(item => ({
        x: item.Open,
        y: item.Close
    }));

    let pointsArr = await points.toArray();
    if(pointsArr.length&1) pointsArr.pop();

    /**
     * Shuffling the data set so that our model does not 
     * encounter similar values in each step
     * */
    tf.util.shuffle(pointsArr)

    plot(pointsArr);

    let features = await pointsArr.map(item => item.x);
    let outputs = await pointsArr.map(item => item.y);
    let featureTensor = tf.tensor2d(features, [features.length,1]);
    let outputTensor = tf.tensor2d(outputs, [outputs.length,1]);
    let normalisedFeatures = normalize(featureTensor);
    let normalisedOutput = normalize(outputTensor);
    let [trainFeatures, testFeatures] = tf.split(normalisedFeatures,2);
    let [trainOutput, testOuput] = tf.split(normalisedOutput,2);

    let model = await load();
    //if model is not present in local storage

    if(!model) {
        model = createModel();
        //tfvis.show.modelSummary({name: 'Summary'}, model);
        let layer = model.getLayer(undefined, 0);
        //tfvis.show.layer({name: 'Layer 1'}, layer);
        const {onEpochEnd} = tfvis.show.fitCallbacks({name: 'Training visuaizing'}, ['loss']);
        const result = await model.fit(trainFeatures, trainOutput, {
            epochs: 10,
            validationData: [trainFeatures, trainOutput],
            callbacks: {
                onEpochEnd
            }
        });

        console.log(result.history.loss.pop());
        // saving the model in local storage so that is not trained everytime the page is refreshed
        save(model);
        const testing =  await model.evaluate(testFeatures, testOuput);
        console.log(await testing.dataSync());
    }

    //Using our model to make predictions
    let normalisedXs = [];
    while(normalisedXs.length <1000){
        var r = Math.random();
        normalisedXs.push(r);
    }
    normalisedXs = tf.tensor2d(normalisedXs, [1000,1])
    const normalisedYs = model.predict(normalisedXs);
    featureTensor.min().print();featureTensor.max().print();
    const xs = denormalize(normalisedXs, featureTensor.min(), featureTensor.max()).dataSync();
    const ys = denormalize(normalisedYs, outputTensor.min(), outputTensor.max()).dataSync();

    const predictedPoints = Array.from(xs).map((val, ind) => ({
        x: val, y: ys[ind]
    }));
    
    plot(pointsArr, predictedPoints);
}

loadAndTrainModel()