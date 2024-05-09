let model; // Variable global para almacenar el modelo

async function loadModel() {
    model = await tf.loadLayersModel('model.json');
}

async function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = function(event) {
            const img = new Image();
            img.onload = function() {
                resolve(img);
            };
            img.onerror = function() {
                reject('Error al cargar la imagen.');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    });
}

async function predict(img) {
    const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();

    const predictions = await model.predict(tensor).data();
    const maxPrediction = Math.max(...predictions);
    const maxIndex = predictions.indexOf(maxPrediction);

    return maxIndex;
}


loadModel();


document.getElementById('imageInput').addEventListener('change', async function(event) {
    const file = event.target.files[0];
    try {
        const img = await loadImage(file);
        const result = await predict(img);
        const resultDiv = document.getElementById('result');
        resultDiv.innerText = `La clase predicha es ${result}.`;
    } catch (error) {
        console.error('Error:', error);
    }
});
