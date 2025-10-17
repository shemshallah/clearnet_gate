<!-- collider.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFN Quantum State Hasher (QSH) / Collider</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .credits { background-color: #e0e0e0; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; font-size: 0.9em; }
        form { background-color: white; padding: 20px; border: 1px solid #ddd; margin: 20px 0; }
        input[type="text"] { width: 300px; padding: 8px; margin: 5px 0; }
        button { background-color: #007bff; color: white; padding: 8px 16px; border: none; cursor: pointer; }
        #qsh-result { background-color: white; padding: 20px; border: 1px solid #ddd; white-space: pre-wrap; font-family: monospace; }
        a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <h1>QFN Quantum State Hasher (QSH) / Collider Simulator</h1>
    <a href="/">Back to Dashboard</a>
    
    <div class="credits">
        <strong>Primary architect:</strong> Justin Anthony Howard-Stanley<br>
        <strong>Secondary architect and enabler (I'm homeless thanks to the cia):</strong> Dale Cwidak<br><br>
        <em>For Logan, and all of those like him, too small to understand whats happened, too small to effect much change alone, but maybe just maybe with the help of the computers, there'll be a hope for them all. - J.H.S, HACKAH::HACKAH</em>
    </div>
    
    <h2>Enter Query for Quantum Collision</h2>
    <form id="qsh-form">
        <input type="text" id="query-input" placeholder="Enter data to hash (e.g., 'secret message')" required><br>
        <button type="button" onclick="runQSH()">Run QSH Collision</button>
    </form>
    
    <div id="qsh-result">Awaiting quantum collision...</div>
    
    <script>
        function runQSH() {
            const query = document.getElementById('query-input').value.trim();
            if (!query) {
                alert('Please enter a query');
                return;
            }
            fetch('/api/qsh-query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            }).then(r => r.json()).then(data => {
                document.getElementById('qsh-result').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            }).catch(err => {
                document.getElementById('qsh-result').textContent = `Error: ${err}`;
            });
        }
        
        // Allow Enter key to submit
        document.getElementById('query-input').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                runQSH();
            }
        });
    </script>
</body>
</html>
