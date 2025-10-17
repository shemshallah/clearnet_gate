<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Collider Interface</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Courier New', monospace; background: #000; color: #0f0; margin: 0; padding: 20px; }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { text-align: center; color: #ff00ff; text-shadow: 0 0 10px #f0f; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .card { background: #111; border: 1px solid #f0f; padding: 15px; border-radius: 5px; box-shadow: 0 0 10px #f0f; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        button { background: #f0f; color: #000; border: none; padding: 10px; cursor: pointer; font-family: inherit; }
        button:hover { background: #ff00ff; box-shadow: 0 0 5px #f0f; }
        canvas { max-width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌌 Quantum Collider - v2.0.0</h1>
        <p style="text-align: center; color: #888;">Timestamp: {{ collider.timestamp }}</p>
        <div class="grid">
            <div class="card">
                <h2>Black Hole</h2>
                <div class="metric"><span>Address:</span><span>{{ collider.black_hole.address }}</span></div>
                <div class="metric"><span>Mass (Solar):</span><span>{{ "%.2e"|format(collider.black_hole.mass_solar) }}</span></div>
                <div class="metric"><span>Event Horizon (km):</span><span>{{ "%.2e"|format(collider.black_hole.event_horizon_km) }}</span></div>
                <div class="metric"><span>Status:</span><span style="color: #0f0;">{{ collider.black_hole.status }}</span></div>
            </div>
            <div class="card">
                <h2>White Hole</h2>
                <div class="metric"><span>Address:</span><span>{{ collider.white_hole.address }}</span></div>
                <div class="metric"><span>Outflow Rate:</span><span>{{ "%.2e"|format(collider.white_hole.outflow_rate) }}</span></div>
                <div class="metric"><span>Status:</span><span style="color: #ff0;">{{ collider.white_hole.status }}</span></div>
            </div>
            <div class="card">
                <h2>Interface</h2>
                <div class="metric"><span>QSH Link:</span><span style="color: #0f0;">{{ collider.interface.qsh_link }}</span></div>
                <div class="metric"><span>Data Rate (Gbps):</span><span>{{ "%.2e"|format(collider.interface.data_rate_gbps) }}</span></div>
                <div class="metric"><span>Entanglement Pairs:</span><span>{{ collider.interface.entanglement_pairs }}</span></div>
                <button onclick="refreshStatus()">Refresh Status</button>
            </div>
        </div>
        <div class="card" style="grid-column: 1 / -1; margin-top: 20px;">
            <h2>Energy Spectrum (Sample)</h2>
            <canvas id="spectrumChart"></canvas>
        </div>
    </div>

    <script>
        // Render initial chart with server data
        const ctx = document.getElementById('spectrumChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({length: {{ collider.energies_ev|length }} }, (_, i) => `Point ${i+1}`),
                datasets: [{
                    label: 'Energy (eV)',
                    data: {{ collider.energies_ev|tojson }},
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    tension: 0.1
                }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });

        async function refreshStatus() {
            const res = await fetch('/api/collider/status');
            const data = await res.json();
            // Update DOM (e.g., document.querySelector('.metric span:last-child').textContent = data.black_hole.status;)
            location.reload();  // Simple refresh for demo
        }
    </script>
</body>
</html>
