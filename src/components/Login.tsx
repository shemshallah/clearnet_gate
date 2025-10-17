import { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

interface Props {
  setToken: (token: string) => void;
}

const Login: React.FC<Props> = ({ setToken }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await axios.post('/api/login', { username, password, remember_me: rememberMe });
      setToken(res.data.access_token);
      localStorage.setItem('token', res.data.access_token);
      navigate('/chat');
    } catch (err) {
      alert('Login failed');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="login-form">
      <h1>Quantum Foam Login</h1>
      <input type="text" placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} />
      <input type="password" placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} />
      <label>
        <input type="checkbox" checked={rememberMe} onChange={(e) => setRememberMe(e.target.checked)} />
        Remember me (saves IP for this device)
      </label>
      <button type="submit">Login</button>
      <p>Your data is protected by black hole hashes from white hole retrieval via quantum colliders.</p>
      <a href="/register">Register</a> | <a href="/forget-password">Forget Password?</a>
    </form>
  );
};

export default Login;