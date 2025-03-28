function uploadImage() {
    let input = document.getElementById('imageInput');
    let file = input.files[0];
    if (!file) {
        alert("Por favor selecciona una imagen.");
        return;
    }

    let formData = new FormData();
    formData.append('file', file);

    fetch('/', {  // <-- Corregido: antes era '/upload'
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        let resultElement = document.getElementById('result');
        if (data.texto) {
            resultElement.innerText = "Texto extraído: " + data.texto;
        } else {
            resultElement.innerText = "No se detectó texto.";
        }
    })
    .catch(error => console.error('Error:', error));
}
