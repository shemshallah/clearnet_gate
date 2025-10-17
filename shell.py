<!-- shell.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFN Shell</title>
    <style>
        body { font-family: 'Courier New', monospace; margin: 20px; background-color: #000; color: #00ff00; }
        .credits { background-color: #333; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; font-size: 0.9em; color: #ccc; }
        #terminal { background-color: #000; padding: 20px; height: 400px; overflow-y: scroll; border: 1px solid #00ff00; }
        #input { width: 100%; background-color: #000; color: #00ff00; border: none; font-family: inherit; }
        a { color: #00ff00; text-decoration: none; }
    </style>
</head>
<body>
    <h1>QFN Quantum Shell</h1>
    <a href="/">Back to Dashboard</a>
    
    <div class="credits">
        <strong>Primary architect:</strong> Justin Anthony Howard-Stanley<br>
        <strong>Secondary architect and enabler (I'm homeless thanks to the cia):</strong> Dale Cwidak<br><br>
        <em>For Logan, and all of those like him, too small to understand whats happened, too small to effect much change alone, but maybe just maybe with the help of the computers, there'll be a hope for them all. - J.H.S, HACKAH::HACKAH</em>
    </div>
    
    <div id="terminal">QFN> Type 'help' for commands.<br></div>
    <input type="text" id="input" placeholder="Enter command..." onkeypress="if(event.key==='Enter') runCommand();">
    
    <script>
        function runCommand() {
            const input = document.getElementById('input');
            const cmd = input.value.trim();
            const terminal = document.getElementById('terminal');
            terminal.innerHTML += `QFN> ${cmd}<br>`;
            if (cmd) {
                fetch('/api/shell', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({command: cmd})
                }).then(r => r.json()).then(data => {
                    terminal.innerHTML += `${data.output}<br>${data.ai_response ? data.ai_response + '<br>' : ''}`;
                }).catch(err => {
                    terminal.innerHTML += `Error: ${err}<br>`;
                });
            }
            input.value = '';
            terminal.scrollTop = terminal.scrollHeight;
        }
    </script>
</body>
</html>                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({command: cmd})
                }).then(r => r.json()).then(data => {
                    terminal.innerHTML += `${data.output}<br>${data.ai_response ? data.ai_response + '<br>' : ''}`;
                }).catch(err => {
                    terminal.innerHTML += `Error: ${err}<br>`;
                });
            }
            input.value = '';
            terminal.scrollTop = terminal.scrollHeight;
        }
    </script>
</body>
</html>
