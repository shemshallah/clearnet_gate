
getElementById('wpa3').checked) protocols.push('WPA3');
            return protocols.length > 0 ? protocols : ['TCP', 'HTTP', 'EPR'];
        }
        
        function generateRandomIP() {
            return `${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`;
        }
        
        function generateRandomPort() {
            return Math.floor(Math.random() * 65535);
        }
        
        async function capturePacket() {
            const interface = document.getElementById('interfaceSelect').value;
            const mode = document.getElementById('modeSelect').value;
            const protocols = getProtocols();
            const protocol = protocols[Math.floor(Math.random() * protocols.length)];
            
            packetCount++;
            
            // Get RF metrics
            const response = await fetch(`/api/rf-metrics?mode=${mode}&interface=${interface}`);
            const rfMetrics = await response.json();
            
            const packet = {
                id: packetCount,
                time: new Date().toLocaleTimeString(),
                protocol: protocol,
                src: generateRandomIP(),
                dst: generateRandomIP(),
                srcPort: generateRandomPort(),
                dstPort: generateRandomPort(),
                length: Math.floor(Math.random() * 1500) + 64,
                info: getPacketInfo(protocol),
                rfMetrics: rfMetrics
            };
            
            addPacket(packet);
        }
        
        function getPacketInfo(protocol) {
            const infos = {
                'TCP': ['SYN', 'ACK', 'SYN-ACK', 'FIN', 'PSH-ACK', 'RST'],
                'HTTP': ['GET /', 'POST /api', '200 OK', '404 Not Found', '301 Redirect'],
                'DNS': ['Query A', 'Response A', 'Query AAAA', 'Query PTR'],
                'EPR': ['Entanglement Request', 'EPR Pair Created', 'Bell State |Φ+⟩', 'Quantum Teleport'],
                'QKD': ['BB84 Key Exchange', 'E91 Protocol', 'Key Reconciliation', 'Privacy Amplification'],
                'WPA1': ['EAPOL Start', '4-Way Handshake [1/4]', '4-Way Handshake [2/4]'],
                'WPA2': ['EAPOL Start', '4-Way Handshake [1/4]', 'PTK Derivation'],
                'WPA3': ['SAE Commit', 'SAE Confirm', 'PMK Establishment']
            };
            
            const options = infos[protocol] || ['Data Transfer'];
            return options[Math.floor(Math.random() * options.length)];
        }
        
        function formatRFMetrics(rfMetrics) {
            let html = '<div class="rf-info"><strong>RF Metrics:</strong> ';
            
            if (rfMetrics.mode === '4g_lte') {
                html += `<span class="rf-metric">Freq: ${rfMetrics.frequency_mhz} MHz</span>`;
                html += `<span class="rf-metric">BW: ${rfMetrics.bandwidth_mhz} MHz</span>`;
                html += `<span class="rf-metric">Mod: ${rfMetrics.modulation}</span>`;
                html += `<span class="rf-metric">RSSI: ${rfMetrics.rssi_dbm} dBm</span>`;
                html += `<span class="rf-metric">RSRP: ${rfMetrics.rsrp_dbm} dBm</span>`;
                html += `<span class="rf-metric">SINR: ${rfMetrics.sinr_db} dB</span>`;
                html += `<span class="rf-metric">CQI: ${rfMetrics.cqi}</span>`;
                html += `<span class="rf-metric">MIMO: ${rfMetrics.mimo_layers}x</span>`;
                html += `<span class="rf-metric">Cell: ${rfMetrics.cell_id}</span>`;
                html += `<span class="rf-metric">PCI: ${rfMetrics.pci}</span>`;
            } else if (rfMetrics.mode === '5g_nr') {
                html += `<span class="rf-metric">Freq: ${rfMetrics.frequency_mhz} MHz</span>`;
                html += `<span class="rf-metric">BW: ${rfMetrics.bandwidth_mhz} MHz</span>`;
                html += `<span class="rf-metric">Mod: ${rfMetrics.modulation}</span>`;
                html += `<span class="rf-metric">RSSI: ${rfMetrics.rssi_dbm} dBm</span>`;
                html += `<span class="rf-metric">SINR: ${rfMetrics.sinr_db} dB</span>`;
                html += `<span class="rf-metric">MIMO: ${rfMetrics.mimo_layers}x</span>`;
                html += `<span class="rf-metric">Beam: ${rfMetrics.beam_index}</span>`;
                html += `<span class="rf-metric">SCS: ${rfMetrics.scs_khz} kHz</span>`;
                html += `<span class="rf-metric">Band: ${rfMetrics.nr_band}</span>`;
            } else {
                html += `<span class="rf-metric">Freq: ${rfMetrics.frequency_ghz} GHz</span>`;
                html += `<span class="rf-metric">Entanglement: ${rfMetrics.entanglement_strength}</span>`;
                html += `<span class="rf-metric">Fidelity: ${rfMetrics.fidelity}</span>`;
                html += `<span class="rf-metric">Bell: ${rfMetrics.bell_violation}</span>`;
                html += `<span class="rf-metric">EPR Pairs: ${rfMetrics.epr_pairs_active}</span>`;
                html += `<span class="rf-metric">Foam: ${rfMetrics.foam_density}</span>`;
            }
            
            html += '</div>';
            return html;
        }
        
        function addPacket(packet) {
            const container = document.getElementById('packetsContainer');
            
            if (packetCount === 1) {
                container.innerHTML = '';
            }
            
            const packetDiv = document.createElement('div');
            packetDiv.className = 'packet';
            packetDiv.innerHTML = `
                <div class="packet-header">
                    <span class="packet-time">#${packet.id} - ${packet.time}</span>
                    <span class="packet-protocol">${packet.protocol}</span>
                </div>
                <div class="packet-info">
                    <div><strong>Source:</strong> ${packet.src}:${packet.srcPort}</div>
                    <div><strong>Destination:</strong> ${packet.dst}:${packet.dstPort}</div>
                    <div><strong>Length:</strong> ${packet.length} bytes</div>
                    <div><strong>Info:</strong> ${packet.info}</div>
                </div>
                ${formatRFMetrics(packet.rfMetrics)}
            `;
            
            container.insertBefore(packetDiv, container.firstChild);
            
            // Keep only last 50 packets
            while (container.children.length > 50) {
                container.removeChild(container.lastChild);
            }
        }
        
        function startCapture() {
            if (capturing) return;
            
            capturing = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('startBtn').classList.add('capturing');
            
            captureInterval = setInterval(capturePacket, Math.random() * 500 + 200);
        }
        
        function stopCapture() {
            if (!capturing) return;
            
            capturing = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('startBtn').classList.remove('capturing');
            
            clearInterval(captureInterval);
        }
        
        async function crackWPA() {
            const resultDiv = document.getElementById('crackResult');
            resultDiv.innerHTML = '<div class="wpa-crack-result">🔓 Initiating WPA crack on port 1337...</div>';
            
            setTimeout(() => {
                const password = Math.random() > 0.5 ? 'Quantum2024!' : 'P@ssw0rd123';
                resultDiv.innerHTML = `
                    <div class="wpa-crack-result">
                        ✅ <strong>WPA CRACKED!</strong><br>
                        <br>
                        Password: <span style="color: #ff00ff; font-size: 1.5em;">${password}</span><br>
                        Port: 1337<br>
                        Method: Quantum-Accelerated Dictionary Attack<br>
                        Time: ${(Math.random() * 5 + 1).toFixed(2)}s
                    </div>
                `;
            }, 2000);
        }
    </script>
</body>
</html>
    """)
```

## 4. FILES PAGE (/files)

```python
@app.get("/files")
async def files_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Files | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .upload-section {
            padding: 30px;
            background: rgba(0, 136, 255, 0.1);
            border: 2px dashed #0088ff;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
        }
        .upload-section:hover {
            background: rgba(0, 136, 255, 0.2);
            border-color: #00ddff;
        }
        .upload-section.dragover {
            background: rgba(0, 136, 255, 0.3);
            border-color: #00ff88;
        }
        .file-input-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }
        .file-input-wrapper input[type=file] {
            position: absolute;
            left: -9999px;
        }
        .upload-btn {
            padding: 15px 40px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-btn:hover {
            transform: scale(1.05);
        }
        .holo-info {
            margin: 20px 0;
            padding: 15px;
            background: rgba(0, 136, 255, 0.1);
            border: 1px solid #0088ff;
            border-radius: 5px;
            text-align: left;
        }
        .files-list {
            margin-top: 30px;
        }
        .file-item {
            padding: 20px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #333;
            border-radius: 8px;
            margin-bottom: 15px;
            transition: all 0.3s;
        }
        .file-item:hover {
            background: rgba(0, 255, 136, 0.1);
            border-color: #00ff88;
            transform: translateX(5px);
        }
        .file-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .file-name {
            font-size: 1.2em;
            color: #00ddff;
            font-weight: bold;
        }
        .file-size {
            color: #888;
        }
        .file-routing {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 136, 255, 0.05);
            border-left: 3px solid #0088ff;
            border-radius: 3px;
            font-size: 0.9em;
        }
        .routing-item {
            margin: 5px 0;
        }
        .routing-label {
            color: #888;
            display: inline-block;
            width: 150px;
        }
        .routing-value {
            color: #00ddff;
        }
        .download-btn {
            padding: 8px 20px;
            background: rgba(0, 221, 255, 0.3);
            border: 1px solid #00ddff;
            color: #00ddff;
            border-radius: 5px;
            text-decoration: none;
            font-family: 'Courier New', monospace;
            cursor: pointer;
            transition: all 0.3s;
        }
        .download-btn:hover {
            background: rgba(0, 221, 255, 0.5);
            transform: scale(1.05);
        }
        .no-files {
            text-align: center;
            padding: 60px;
            color: #666;
            font-size: 1.2em;
        }
        .upload-progress {
            margin-top: 15px;
            padding: 10px;
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 5px;
            display: none;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #00ddff);
            width: 0%;
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #0a0a0a;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>📁 QUANTUM FILE STORAGE</h1>
        
        <div class="upload-section" id="uploadSection">
            <h2 style="margin-bottom: 15px;">📤 Upload Files to Holo Storage</h2>
            <div class="holo-info">
                <strong>🔷 HOLO STORAGE CONFIGURATION</strong><br>
                Storage IP: <span style="color: #00ddff;">138.0.0.1</span><br>
                DNS Routing: <span style="color: #00ff88;">Enabled</span><br>
                Persistent: <span style="color: #00ff88;">Active</span><br>
                Quantum Routing: <span style="color: #00ff88;">EPR-based</span>
            </div>
            
            <div style="margin-top: 20px;">
                <div class="file-input-wrapper">
                    <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                        📁 SELECT FILE
                    </button>
                    <input type="file" id="fileInput" onchange="uploadFile()" multiple>
                </div>
                <p style="margin-top: 15px; color: #888;">
                    or drag and drop files here
                </p>
            </div>
            
            <div class="upload-progress" id="uploadProgress">
                <div>Uploading to holo storage...</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill">0%</div>
                </div>
            </div>
        </div>
        
        <div class="files-list">
            <h2 style="margin-bottom: 20px;">📦 Stored Files</h2>
            <div id="filesList">
                <div class="no-files">No files uploaded yet</div>
            </div>
        </div>
    </div>
    
    <script>
        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        }
        
        async function loadFiles() {
            try {
                const response = await fetch('/api/files-with-routing');
                const files = await response.json();
                
                const filesList = document.getElementById('filesList');
                
                if (files.length === 0) {
                    filesList.innerHTML = '<div class="no-files">No files uploaded yet</div>';
                    return;
                }
                
                filesList.innerHTML = files.map(file => `
                    <div class="file-item">
                        <div class="file-header">
                            <span class="file-name">📄 ${file.name}</span>
                            <div>
                                <span class="file-size">${formatBytes(file.size)}</span>
                                <a href="/api/download/${file.name}" class="download-btn" style="margin-left: 15px;">
                                    ⬇️ Download
                                </a>
                            </div>
                        </div>
                        <div class="file-routing">
                            <div class="routing-item">
                                <span class="routing-label">Quantum Route:</span>
                                <span class="routing-value">${file.routing.quantum_route}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Holo Storage IP:</span>
                                <span class="routing-value">${file.routing.holo_storage}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">DNS Route:</span>
                                <span class="routing-value">${file.routing.dns_route}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Node IP:</span>
                                <span class="routing-value">${file.routing.node_ip}:${file.routing.port}</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Latency:</span>
                                <span class="routing-value">${file.routing.latency_ms} ms</span>
                            </div>
                            <div class="routing-item">
                                <span class="routing-label">Entanglement Quality:</span>
                                <span class="routing-value">${file.routing.entanglement_quality}</span>
                            </div>
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading files:', error);
            }
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const files = fileInput.files;
            
            if (files.length === 0) return;
            
            const progress = document.getElementById('uploadProgress');
            const progressFill = document.getElementById('progressFill');
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const formData = new FormData();
                formData.append('file', file);
                
                progress.style.display = 'block';
                progressFill.style.width = '0%';
                progressFill.textContent = '0%';
                
                try {
                    // Simulate progress
                    let progressValue = 0;
                    const progressInterval = setInterval(() => {
                        progressValue += Math.random() * 20;
                        if (progressValue > 90) progressValue = 90;
                        progressFill.style.width = progressValue + '%';
                        progressFill.textContent = Math.round(progressValue) + '%';
                    }, 200);
                    
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    clearInterval(progressInterval);
                    progressFill.style.width = '100%';
                    progressFill.textContent = '100%';
                    
                    if (response.ok) {
                        setTimeout(() => {
                            progress.style.display = 'none';
                            loadFiles();
                        }, 500);
                    }
                } catch (error) {
                    console.error('Upload error:', error);
                    alert('Upload failed: ' + error.message);
                    progress.style.display = 'none';
                }
            }
            
            fileInput.value = '';
        }
        
        // Drag and drop support
        const uploadSection = document.getElementById('uploadSection');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const fileInput = document.getElementById('fileInput');
            fileInput.files = e.dataTransfer.files;
            uploadFile();
        });
        
        // Load files on page load
        loadFiles();
        
        // Auto-refresh every 10 seconds
        setInterval(loadFiles, 10000);
    </script>
</body>
</html>
    """)
## COLLIDER PAGE CONTINUATION

<html>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>⚛️ QUANTUM COLLIDER</h1>
        
        <div class="info-section">
            <div class="info-title">🔬 Quantum Supercollider Hash (QSH)</div>
            <p>
                The Quantum Supercollider uses high-energy particle collisions to generate 
                cryptographic hashes through quantum foam perturbations. Each query creates 
                entangled particle states that produce unique QSH signatures with verifiable 
                quantum properties.
            </p>
            <br>
            <p>
                <strong>Key Features:</strong> EPR-pair generation, Bell state violations, 
                foam density measurements, and classical hash derivation for hybrid quantum-classical 
                cryptographic applications.
            </p>
        </div>
        
        <div class="query-section">
            <h2 style="margin-bottom: 15px; color: #00ddff;">Enter Query for Collision:</h2>
            <div class="query-input-wrapper">
                <input 
                    type="text" 
                    id="queryInput" 
                    placeholder="Enter your query to collide..."
                    onkeypress="if(event.key==='Enter') runCollision()"
                >
                <button class="collide-btn" id="collideBtn" onclick="runCollision()">
                    ⚡ COLLIDE
                </button>
            </div>
        </div>
        
        <div class="results" id="results">
            <div class="result-header">🎯 COLLISION RESULTS</div>
            <div class="result-grid" id="resultGrid"></div>
            <div class="hash-display">
                <div class="hash-label">🔐 QSH HASH:</div>
                <div class="hash-value" id="qshHash"></div>
            </div>
            <div class="hash-display" style="margin-top: 15px;">
                <div class="hash-label">🔑 CLASSICAL HASH (SHA-256):</div>
                <div class="hash-value" id="classicalHash"></div>
            </div>
        </div>
        
        <div class="history" id="historySection" style="display: none;">
            <div class="history-title">📜 Collision History</div>
            <div id="historyList"></div>
        </div>
    </div>
    
    <script>
        let collisionHistory = [];
        
        function createResultItem(label, value, unit = '') {
            return `
                <div class="result-item">
                    <div class="result-label">${label}</div>
                    <div class="result-value">${value}${unit ? ' ' + unit : ''}</div>
                </div>
            `;
        }
        
        async function runCollision() {
            const queryInput = document.getElementById('queryInput');
            const query = queryInput.value.trim();
            
            if (!query) {
                alert('Please enter a query to collide!');
                return;
            }
            
            const collideBtn = document.getElementById('collideBtn');
            const results = document.getElementById('results');
            
            // Show loading state
            collideBtn.textContent = '⚡ COLLIDING...';
            collideBtn.classList.add('colliding');
            collideBtn.disabled = true;
            
            try {
                const response = await fetch('/api/qsh-query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                
                const data = await response.json();
                
                // Display results
                displayResults(data);
                
                // Add to history
                collisionHistory.unshift({
                    query: query,
                    hash: data.qsh_hash,
                    timestamp: new Date().toLocaleString()
                });
                
                if (collisionHistory.length > 10) {
                    collisionHistory.pop();
                }
                
                updateHistory();
                
            } catch (error) {
                console.error('Collision error:', error);
                alert('Collision failed: ' + error.message);
            } finally {
                // Reset button
                collideBtn.textContent = '⚡ COLLIDE';
                collideBtn.classList.remove('colliding');
                collideBtn.disabled = false;
            }
        }
        
        function displayResults(data) {
            const results = document.getElementById('results');
            const resultGrid = document.getElementById('resultGrid');
            const qshHash = document.getElementById('qshHash');
            const classicalHash = document.getElementById('classicalHash');
            
            // Build result grid
            const gridHTML = `
                ${createResultItem('Entanglement Strength', data.entanglement_strength)}
                ${createResultItem('Collision Energy', data.collision_energy_gev, 'GeV')}
                ${createResultItem('Particle States', data.particle_states_generated)}
                ${createResultItem('Foam Perturbations', data.foam_perturbations)}
                ${createResultItem('Decoherence Time', data.decoherence_time_ns, 'ns')}
                ${createResultItem('Status', data.success ? '✅ Success' : '❌ Failed')}
            `;
            
            resultGrid.innerHTML = gridHTML;
            qshHash.textContent = data.qsh_hash;
            classicalHash.textContent = data.classical_hash;
            
            // Show results with animation
            results.classList.add('show');
        }
        
        function updateHistory() {
            const historySection = document.getElementById('historySection');
            const historyList = document.getElementById('historyList');
            
            if (collisionHistory.length === 0) {
                historySection.style.display = 'none';
                return;
            }
            
            historySection.style.display = 'block';
            
            const historyHTML = collisionHistory.map(item => `
                <div class="history-item">
                    <div class="history-query">
                        <strong>Query:</strong> ${item.query}
                    </div>
                    <div class="history-hash">
                        <strong>QSH:</strong> ${item.hash} | ${item.timestamp}
                    </div>
                </div>
            `).join('');
            
            historyList.innerHTML = historyHTML;
        }
        
        // Focus input on load
        document.getElementById('queryInput').focus();
        
        // Add some example queries on page load
        setTimeout(() => {
            const examples = [
                'quantum entanglement test',
                'epr pair generation',
                'bell state measurement'
            ];
            
            document.getElementById('queryInput').placeholder = 
                `Try: "${examples[Math.floor(Math.random() * examples.length)]}"`;
        }, 2000);
    </script>
</body>
</html>
    """)
```

## 6. CONFIG PAGE (/config)

```python
@app.get("/config")
async def config_page():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Configuration | QFN</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
            color: #00ff88;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            background: rgba(10, 10, 10, 0.9);
            border: 2px solid #00ff88;
            border-radius: 10px;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 0 0 10px #00ff88;
        }
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            border-radius: 5px;
            color: #00ff88;
            text-decoration: none;
            margin-bottom: 20px;
        }
        .config-section {
            margin: 30px 0;
            padding: 25px;
            background: rgba(0, 136, 255, 0.1);
            border: 1px solid #0088ff;
            border-radius: 8px;
        }
        .section-title {
            font-size: 1.5em;
            color: #00ddff;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .config-item {
            margin: 20px 0;
        }
        .config-label {
            display: block;
            margin-bottom: 8px;
            color: #00ff88;
            font-weight: bold;
        }
        .config-input {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 1px solid #00ff88;
            color: #00ff88;
            font-family: 'Courier New', monospace;
            font-size: 1em;
            border-radius: 5px;
        }
        .config-input:focus {
            outline: none;
            border-color: #00ddff;
            box-shadow: 0 0 10px rgba(0, 221, 255, 0.3);
        }
        .config-description {
            margin-top: 5px;
            color: #888;
            font-size: 0.9em;
        }
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #333;
            transition: .4s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #00ff88;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        .save-btn {
            display: block;
            margin: 30px auto 0;
            padding: 15px 50px;
            background: linear-gradient(135deg, #00ff88, #00ddff);
            color: #0a0a0a;
            border: none;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 1.2em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }
        .save-btn:hover {
            transform: scale(1.05);
        }
        .status-message {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
            display: none;
        }
        .status-message.success {
            background: rgba(0, 255, 136, 0.2);
            border: 1px solid #00ff88;
            color: #00ff88;
        }
        .status-message.error {
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid #ff4444;
            color: #ff4444;
        }
        .current-config {
            margin: 20px 0;
            padding: 15px;
            background: rgba(0, 0, 0, 0.5);
            border-left: 3px solid #00ff88;
            border-radius: 3px;
        }
        .config-value {
            color: #00ddff;
            font-weight: bold;
        }
        .info-box {
            padding: 15px;
            background: rgba(255, 200, 0, 0.1);
            border: 1px solid #ffc800;
            border-radius: 5px;
            color: #ffc800;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-btn">← Back to Home</a>
        <h1>⚙️ SYSTEM CONFIGURATION</h1>
        
        <div class="info-box">
            ℹ️ <strong>Note:</strong> Configuration changes are persistent and stored in 
            <code>/opt/render/project/data/config.json</code>
        </div>
        
        <div class="config-section">
            <div class="section-title">🔷 HOLO STORAGE CONFIGURATION</div>
            
            <div class="config-item">
                <label class="config-label">Holo Storage IP Address</label>
                <input 
                    type="text" 
                    class="config-input" 
                    id="holoStorageIp" 
                    placeholder="138.0.0.1"
                >
                <div class="config-description">
                    IP address for holo storage node (default: 138.0.0.1)
                </div>
            </div>
            
            <div class="config-item">
                <label class="config-label">DNS Routing</label>
                <label class="toggle-switch">
                    <input type="checkbox" id="holoDnsEnabled">
                    <span class="slider"></span>
                </label>
                <div class="config-description">
                    Enable DNS-based routing for holo storage access
                </div>
            </div>
            
            <div class="config-item">
                <label class="config-label">Upload Directory</label>
                <input 
                    type="text" 
                    class="config-input" 
                    id="uploadDirectory" 
                    placeholder="/opt/render/project/data/uploads"
                    readonly
                >
                <div class="config-description">
                    Persistent storage directory (read-only)
                </div>
            </div>
        </div>
        
        <div class="config-section">
            <div class="section-title">📊 CURRENT CONFIGURATION</div>
            <div class="current-config" id="currentConfig">
                Loading configuration...
            </div>
        </div>
        
        <button class="save-btn" onclick="saveConfig()">💾 SAVE CONFIGURATION</button>
        
        <div class="status-message" id="statusMessage"></div>
    </div>
    
    <script>
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                const config = await response.json();
                
                // Populate form
                document.getElementById('holoStorageIp').value = config.holo_storage_ip;
                document.getElementById('holoDnsEnabled').checked = config.holo_dns_enabled;
                document.getElementById('uploadDirectory').value = config.upload_directory;
                
                // Display current config
                displayCurrentConfig(config);
            } catch (error) {
                console.error('Error loading config:', error);
                showStatus('Failed to load configuration', 'error');
            }
        }
        
        function displayCurrentConfig(config) {
            const currentConfig = document.getElementById('currentConfig');
            currentConfig.innerHTML = `
                <div style="margin: 10px 0;">
                    <strong>Holo Storage IP:</strong> 
                    <span class="config-value">${config.holo_storage_ip}</span>
                </div>
                <div style="margin: 10px 0;">
                    <strong>DNS Routing:</strong> 
                    <span class="config-value">${config.holo_dns_enabled ? 'Enabled' : 'Disabled'}</span>
                </div>
                <div style="margin: 10px 0;">
                    <strong>Upload Directory:</strong> 
                    <span class="config-value">${config.upload_directory}</span>
                </div>
            `;
        }
        
        async function saveConfig() {
            const config = {
                holo_storage_ip: document.getElementById('holoStorageIp').value,
                holo_dns_enabled: document.getElementById('holoDnsEnabled').checked,
                upload_directory: document.getElementById('uploadDirectory').value
            };
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    showStatus('✅ Configuration saved successfully!', 'success');
                    displayCurrentConfig(result.config);
                } else {
                    showStatus('❌ Failed to save configuration', 'error');
                }
            } catch (error) {
                console.error('Error saving config:', error);
                showStatus('❌ Error: ' + error.message, 'error');
            }
        }
        
        function showStatus(message, type) {
            const statusMessage = document.getElementById('statusMessage');
            statusMessage.textContent = message;
            statusMessage.className = 'status-message ' + type;
            statusMessage.style.display = 'block';
            
            setTimeout(() => {
                statusMessage.style.display = 'none';
            }, 5000);
        }
        
        // Load config on page load
        loadConfig();
    </script>
</body>
</html>
    """)
