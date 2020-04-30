const tf = require('@tensorflow/tfjs');

/**
 *   This script is used to load a saved model and perform inference.
 *   Run this script in console:
 *   ./index MODEL_URL INPUT_DATA MODEL_TYPE
 *   
 *   MODEL_TYPE: graph_model | layers_model = graph_model
 *   INPUT_DATA: 0 for default inputs.
 */
async function main() {
    const args = process.argv.slice(2);

    if (args.length < 2) {
        console.log('You need to pass model url and input data');
        return 'fail';
    }

    const modelUrl = args[0];
    const modelType = parseModelType(args[2]);
    const defaultInputs = tf.zeros([1, 224, 224, 3], 'int32');
    const inputs = args[1] === '0' ? defaultInputs : args[1];

    let model;
    let result;

    if (modelType === 'graph_model') {
        model = await tf.loadGraphModel(modelUrl);
        result = await model.executeAsync(inputs);
    } else if (modelType === 'layers_model') {
        model = await tf.loadLayersModel(modelUrl);
        result = model.predict(inputs);
    } else {
        return 'fail';
    }

    console.log(result);

    defaultInputs.dispose();

    return 'done';
}

function parseModelType(modelType) {
    if (!modelType) {
        return 'graph_model';
    }

    if ((modelType !== 'graph_model') && (modelType !== 'layers_model')) {
        console.log(`Unhandled model type: ${modelType}. Please use a ` +
        `valid type: graph_model, layers_model`);
        return null;
    }

    return modelType;
}

main();
