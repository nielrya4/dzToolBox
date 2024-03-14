document.addEventListener("DOMContentLoaded", function() {
    var loadingBar = document.getElementById('spinner');
    var loadButton = document.getElementById('generate_outputs');
    loadingBar.classList.add('hidden');

    loadButton.addEventListener('click', function() {
        loadingBar.classList.remove('hidden');
        loadingBar.classList.add('visible');

        // Hide loading bar after a delay (simulated data loading)
        setTimeout(function() {
            loadingBar.classList.add('hidden');
            loadingBar.classList.remove('visible');
        }, 50000);
    });
});
