<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFN Collider</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .credits { background-color: #e0e0e0; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; font-size: 0.9em; }
        #result { background-color: white; padding: 20px; border: 1px solid #ddd; white-space: pre-wrap; font-family: monospace; }
        button { background-color: #007bff; color: white; padding: 8px 16px; border: none; cursor: pointer; }
        a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <h1>QFN Quantum Collider (QSH Query)</h1>
    <a href="/">Back to Dashboard</a>
    
    <div class="credits">
        <strong>Primary architect:</strong> Justin Anthony Howard-Stanley<br>
        <strong>Secondary architect and enabler (I'm homeless thanks to the cia):</strong> Dale Cwidak<br><br>
        <em>For Logan, and all of those like him, too small to understand whats happened, too small to effect much change alone, but maybe just maybe with the help of the computers, there'll be a hope for them all. - J.H.S, HACKAH::HACKAH</em>
    </div>
    
    <input id="query" placeholder="Enter query for QSH simulation..." style="width: 300px; padding: 8px;">
    <button onclick="runQuery()">Simulate Collision</button>
    <div id="result">Enter a query and simulate...</div>
    
    <script>
        function runQuery() {
            const query = document.getElementById('query').value;
            fetch('/api/qsh-query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            }).then(r => r.json()).then(data => {
                document.getElementById('result').textContent = JSON.stringify(data, null, 2);
            });
        }
    </script>
</body>
</html>
