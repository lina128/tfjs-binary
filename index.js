const tf = require('@tensorflow/tfjs');

async function main() {
    // How to decide loadGraphModel or loadLayersModel or loadGraphModel with hub flag true.
    const model = await tf.loadGraphModel('https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json');

    const zeros = tf.zeros([1, 224, 224, 3]);

    // How to decide executeAsync or execute or predict.
    const result = await model.executeAsync(zeros);

    console.log(result);

    zeros.dispose();

    return 'done';
}

main();
