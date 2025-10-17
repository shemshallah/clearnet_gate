<!DOCTYPE html>
<html>
<head>
    <title>Quantum Chat</title>
    <style> /* Dark theme */ </style>
</head>
<body>
    <h1>Entangled Chat</h1>
    <div id="messages"></div>
    <input id="messageInput" type="text" placeholder="Type message...">
    <button onclick="sendMessage()">Send</button>
    <div id="matches">Matches: {{ matches|join(', ') }}</div>
    <script>
        const ws = new WebSocket(`ws://localhost:8000/ws/{{ user.id }}`);
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            document.getElementById('messages').innerHTML += `<p>${data.content}</p>`;
        };
        function sendMessage() {
            const input = document.getElementById('messageInput');
            ws.send(JSON.stringify({content: input.value, receiver_id: 0}));  // 0 for broadcast
            input.value = '';
        }
        // AI chat button
        function sendToAI() {
            ws.send(JSON.stringify({content: input.value, is_ai: true}));
        }
    </script>
</body>
</html>
