const tf = require('@tensorflow/tfjs');

async function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log('You need to pass model url as the first argument');
        return 'fail';
    }

    const modelUrl = args[0];

    // How to decide loadGraphModel or loadLayersModel or loadGraphModel with hub flag true.
    const model = await tf.loadGraphModel(modelUrl);

    const zeros = tf.zeros([1, 224, 224, 3], 'int32');

    // How to decide executeAsync or execute or predict.
    const result = await model.executeAsync(zeros);

    console.log(result);

    zeros.dispose();

    return 'done';
}

main();
