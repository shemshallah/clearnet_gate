import { useEffect, useState } from 'react';
import axios from 'axios';

interface Props {
  token: string;
}

const Inbox: React.FC<Props> = ({ token }) => {
  const [emails, setEmails] = useState([]);
  const [search, setSearch] = useState('');
  const [folder, setFolder] = useState('Inbox');
  const [newSubject, setNewSubject] = useState('');
  const [newBody, setNewBody] = useState('');
  const [newReceiver, setNewReceiver] = useState('');

  useEffect(() => {
    fetchInbox();
  }, [search, folder]);

  const fetchInbox = async () => {
    const res = await axios.get(`/api/inbox?search=${search}&folder=${folder}`);
    setEmails(res.data);
  };

  const sendEmail = async (e: React.FormEvent) => {
    e.preventDefault();
    await axios.post('/api/inbox/send', { receiver_email: newReceiver, subject: newSubject, body: newBody });
    setNewSubject(''); setNewBody(''); setNewReceiver('');
    fetchInbox();
  };

  const deleteEmail = async (id: number) => {
    await axios.delete(`/api/inbox/${id}`);
    fetchInbox();
  };

  const starEmail = async (id: number) => {
    await axios.post(`/api/inbox/${id}/star`);
    fetchInbox();
  };

  return (
    <div className="inbox">
      <h1>Holo Inbox - Searchable Forever</h1>
      <input placeholder="Search emails..." value={search} onChange={(e) => setSearch(e.target.value)} />
      <select value={folder} onChange={(e) => setFolder(e.target.value)}>
        <option>Inbox</option>
        <option>Sent</option>
        <option>Archive</option>
        <option>Spam</option>
      </select>
      <div className="emails">
        {emails.map((email: any) => (
          <div key={email.id} className="email-item">
            <h3>{email.subject} <small>{email.timestamp}</small></h3>
            <p>{email.body}...</p>
            <small>Labels: {email.label} | Secured by collider black hole hash</small>
            <button onClick={() => starEmail(email.id)}>{email.is_starred ? 'Unstar' : 'Star'}</button>
            <button onClick={() => deleteEmail(email.id)}>Delete</button>
          </div>
        ))}
      </div>
      <form onSubmit={sendEmail}>
        <input placeholder="To (email@foam.computer)" value={newReceiver} onChange={(e) => setNewReceiver(e.target.value)} />
        <input placeholder="Subject" value={newSubject} onChange={(e) => setNewSubject(e.target.value)} />
        <textarea placeholder="Body" value={newBody} onChange={(e) => setNewBody(e.target.value)} />
        <button type="submit">Send Encrypted Email</button>
      </form>
      <p>Import contacts: <input type="file" onChange={(e) => {/* Handle CSV */}} /></p>
    </div>
  );
};

export default Inbox;