<!-- files.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFN Files</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .credits { background-color: #e0e0e0; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        a { color: #007bff; text-decoration: none; }
        button { background-color: #007bff; color: white; padding: 8px 16px; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <h1>QFN File Management</h1>
    <a href="/">Back to Dashboard</a>
    
    <div class="credits">
        <strong>Primary architect:</strong> Justin Anthony Howard-Stanley<br>
        <strong>Secondary architect and enabler (I'm homeless thanks to the cia):</strong> Dale Cwidak<br><br>
        <em>For Logan, and all of those like him, too small to understand whats happened, too small to effect much change alone, but maybe just maybe with the help of the computers, there'll be a hope for them all. - J.H.S, HACKAH::HACKAH</em>
    </div>
    
    <button onclick="loadFiles()">Refresh Files</button>
    <table id="files-table">
        <thead><tr><th>Name</th><th>Size</th><th>Quantum Route</th><th>Holo IP</th><th>Latency</th><th>Actions</th></tr></thead>
        <tbody></tbody>
    </table>
    
    <script>
        function loadFiles() {
            fetch('/api/files-with-routing?limit=20').then(r => r.json()).then(files => {
                const tbody = document.querySelector('#files-table tbody');
                tbody.innerHTML = '';
                files.forEach(f => {
                    const tr = tbody.insertRow();
                    tr.insertCell(0).textContent = f.name;
                    tr.insertCell(1).textContent = f.size + ' bytes';
                    tr.insertCell(2).textContent = f.routing.quantum_route;
                    tr.insertCell(3).textContent = f.routing.holo_storage;
                    tr.insertCell(4).textContent = f.routing.latency_ms + ' ms';
                    const actions = tr.insertCell(5);
                    actions.innerHTML = `<a href="/api/download/${f.name}">Download</a>`;
                });
            });
        }
        loadFiles();  // Initial load
    </script>
</body>
</html>
