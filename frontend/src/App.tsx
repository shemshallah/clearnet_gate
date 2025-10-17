import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import Chatroom from './components/Chatroom';
import Inbox from './components/Inbox';
import { useState } from 'react';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));

  return (
    <Router>
      <div className="app dark-theme">
        <Routes>
          <Route path="/" element={<Login setToken={setToken} />} />
          <Route path="/register" element={<Register setToken={setToken} />} />
          <Route path="/chat" element={token ? <Chatroom token={token} /> : <Login />} />
          <Route path="/inbox" element={token ? <Inbox token={token} /> : <Login />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;