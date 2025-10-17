<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFN Metrics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f0f0f0; }
        .credits { background-color: #e0e0e0; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; font-size: 0.9em; }
        #metrics { background-color: white; padding: 20px; border: 1px solid #ddd; white-space: pre-wrap; font-family: monospace; }
        button { background-color: #007bff; color: white; padding: 8px 16px; border: none; cursor: pointer; margin: 5px; }
        a { color: #007bff; text-decoration: none; }
    </style>
</head>
<body>
    <h1>QFN Metrics Dashboard</h1>
    <a href="/">Back to Dashboard</a>
    
    <div class="credits">
        <strong>Primary architect:</strong> Justin Anthony Howard-Stanley<br>
        <strong>Secondary architect and enabler (I'm homeless thanks to the cia):</strong> Dale Cwidak<br><br>
        <em>For Logan, and all of those like him, too small to understand whats happened, too small to effect much change alone, but maybe just maybe with the help of the computers, there'll be a hope for them all. - J.H.S, HACKAH::HACKAH</em>
    </div>
    
    <h2>Current Metrics</h2>
    <div id="metrics">Loading metrics...</div>
    <button onclick="loadMetrics()">Refresh Metrics</button>
    <button onclick="runSpeedTest()">Run Speed Test</button>
    <div id="speed-result"></div>
    
    <h2>RF Metrics</h2>
    <select id="rf-mode">
        <option value="quantum">Quantum</option>
        <option value="4g_lte">4G LTE</option>
        <option value="5g_nr">5G NR</option>
    </select>
    <button onclick="loadRFMetrics()">Load RF Metrics</button>
    <div id="rf-metrics"></div>
    
    <script>
        function loadMetrics() {
            fetch('/api/metrics').then(r => r.json()).then(data => {
                document.getElementById('metrics').textContent = JSON.stringify(data, null, 2);
            });
        }
        loadMetrics();  // Initial load
        
        function runSpeedTest() {
            fetch('/api/speed-test').then(r => r.json()).then(data => {
                document.getElementById('speed-result').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            });
        }
        
        function loadRFMetrics() {
            const mode = document.getElementById('rf-mode').value;
            fetch(`/api/rf-metrics?mode=${mode}`).then(r => r.json()).then(data => {
                document.getElementById('rf-metrics').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            });
        }
    </script>
</body>
</html>
