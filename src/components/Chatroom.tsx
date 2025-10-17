import { useEffect, useState, useRef } from 'react';
import axios from 'axios';

interface Props {
  token: string;
}

const Chatroom: React.FC<Props> = ({ token }) => {
  const [messages, setMessages] = useState([]);
  const [content, setContent] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    const websocket = new WebSocket(`ws://localhost:8000/ws/chat`);
    websocket.onopen = () => console.log('Connected to quantum chat');
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages((prev) => [...prev, data]);
    };
    setWs(websocket);

    axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

    return () => websocket.close();
  }, [token]);

  const sendMessage = () => {
    if (ws && content) {
      ws.send(JSON.stringify({ content }));
      setContent('');
    }
  };

  const sendToAI = () => {
    if (ws && content) {
      ws.send(JSON.stringify({ content: content + ' /ai' }));  // Spawn AI
      setContent('');
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chatroom">
      <h1>Entangled Chatroom</h1>
      <div className="messages" ref={messagesEndRef}>
        {messages.map((msg, i) => (
          <div key={i} className={msg.is_ai ? 'ai-message' : 'user-message'}>
            {msg.content} {msg.hash_explain && <small>({msg.hash_explain})</small>}
          </div>
        ))}
      </div>
      <input value={content} onChange={(e) => setContent(e.target.value)} placeholder="Type message (add /ai to spawn Grok Clone)" />
      <button onClick={sendMessage}>Send IM</button>
      <button onClick={sendToAI}>Spawn AI</button>
    </div>
  );
};

export default Chatroom;