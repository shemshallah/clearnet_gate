<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Foam Email System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Courier New', monospace;
            background: linear-gradient(135deg, #001a33, #003366, #001a33);
            color: #e0e0e0;
            padding: 20px;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 30px;
            border: 2px solid #00ffff;
            background: rgba(0, 255, 255, 0.1);
            margin-bottom: 30px;
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.3);
        }
        
        h1 {
            color: #00ffff;
            text-shadow: 0 0 20px #00ffff;
            font-size: 3em;
            margin-bottom: 10px;
        }
        
        .back-link {
            display: inline-block;
            padding: 10px 20px;
            background: #00ffff;
            color: #000;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 20px;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .back-link:hover {
            background: #00cccc;
            box-shadow: 0 0 20px #00ffff;
        }
        
        .email-grid {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
            margin: 30px 0;
        }
        
        .sidebar {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #00ffff;
            padding: 20px;
            height: fit-content;
        }
        
        .sidebar h2 {
            color: #00ffff;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .user-info {
            background: rgba(0, 255, 255, 0.1);
            padding: 15px;
            margin-bottom: 20px;
            border-left: 3px solid #00ffff;
        }
        
        .user-email {
            color: #ffff00;
            font-weight: bold;
            margin-top: 10px;
            word-wrap: break-word;
        }
        
        .nav-buttons {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .nav-btn {
            padding: 12px;
            background: rgba(0, 255, 255, 0.2);
            border: 2px solid #00ffff;
            color: #00ffff;
            cursor: pointer;
            transition: all 0.3s;
            text-align: left;
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        
        .nav-btn:hover {
            background: rgba(0, 255, 255, 0.3);
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        }
        
        .nav-btn.active {
            background: #00ffff;
            color: #000;
        }
        
        .main-content {
            background: rgba(0, 0, 0, 0.7);
            border: 2px solid #00ffff;
            padding: 30px;
            min-height: 600px;
        }
        
        .content-section {
            display: none;
        }
        
        .content-section.active {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .section-header {
            color: #00ffff;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #00ffff;
            font-size: 2em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            color: #00ffff;
            margin-bottom: 8px;
            font-weight: bold;
        }
        
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 12px;
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #00ffff;
            border-radius: 5px;
            color: #e0e0e0;
            font-family: 'Courier New', monospace;
            font-size: 1em;
        }
        
        .form-group input:focus,
        .form-group textarea:focus {
            outline: none;
            box-shadow: 0 0 15px #00ffff;
        }
        
        .form-group textarea {
            min-height: 200px;
            resize: vertical;
        }
        
        .btn-primary {
            padding: 15px 30px;
            background: #00ffff;
            border: none;
            border-radius: 5px;
            color: #000;
            font-weight: bold;
            font-size: 1.1em;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            transition: all 0.3s;
        }
        
        .btn-primary:hover {
            background: #00cccc;
            box-shadow: 0 0 25px #00ffff;
        }
        
        .email-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .email-item {
            background: rgba(0, 255, 255, 0.1);
            border: 2px solid #00ffff;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .email-item:hover {
            background: rgba(0, 255, 255, 0.2);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
        }
        
        .email-item.unread {
            border-left: 5px solid #ffff00;
            background: rgba(255, 255, 0, 0.1);
        }
        
        .email-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .email-from {
            color: #00ffff;
            font-weight: bold;
        }
        
        .email-date {
            color: #888;
            font-size: 0.9em;
        }
        
        .email-subject {
            color: #ffff00;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .email-preview {
            color: #e0e0e0;
            font-size: 0.95em;
        }
        
        .email-detail {
            background: rgba(0, 0, 0, 0.5);
            border: 2px solid #00ff00;
            padding: 30px;
            margin-top: 20px;
        }
        
        .email-detail-header {
            margin-bottom: 20px;
            padding-bottom: 20px;
            border-bottom: 2px solid #00ff00;
        }
        
        .email-detail-subject {
            color: #00ff00;
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        
        .email-detail-meta {
            color: #00ffff;
            margin: 5px 0;
        }
        
        .email-detail-body {
            color: #e0e0e0;
            line-height: 1.8;
            white-space: pre-wrap;
        }
        
        .btn-back {
            padding: 10px 20px;
            background: rgba(0, 255, 255, 0.2);
            border: 2px solid #00ffff;
            color: #00ffff;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .btn-back:hover {
            background: rgba(0, 255, 255, 0.3);
        }
        
        .success-message {
            background: rgba(0, 255, 0, 0.2);
            border: 2px solid #00ff00;
            color: #00ff00;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            display: none;
        }
        
        .success-message.visible {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        .error-message {
            background: rgba(255, 0, 0, 0.2);
            border: 2px solid #ff0000;
            color: #ff0000;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            display: none;
        }
        
        .error-message.visible {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #888;
        }
        
        .empty-state-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        
        @media (max-width: 768px) {
            h1 { font-size: 2em; }
            .email-grid {
                grid-template-columns: 1fr;
            }
            .sidebar {
                margin-bottom: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <a href="/" class="back-link">← Back to Main</a>
        
        <div class="header">
            <h1>📧 QUANTUM FOAM EMAIL SYSTEM</h1>
            <p style="font-size: 1.2em; margin-top: 10px; color: #00ffff;">
                username::quantum.foam addressing
            </p>
        </div>
        
        <div class="email-grid">
            <!-- Sidebar -->
            <div class="sidebar">
                <h2>Email Client</h2>
                
                <div class="user-info">
                    <div style="color: #00ffff;">Logged in as:</div>
                    <div id="currentUser" style="color: #ffff00; font-weight: bold;">guest</div>
                    <div class="user-email" id="userEmail">guest::quantum.foam</div>
                </div>
                
                <div class="nav-buttons">
                    <button class="nav-btn active" data-view="inbox">📥 Inbox</button>
                    <button class="nav-btn" data-view="compose">✉️ Compose</button>
                </div>
                
                <div style="margin-top: 30px; padding: 15px; background: rgba(255, 255, 0, 0.1); border-left: 3px solid #ffff00;">
                    <strong style="color: #ffff00;">Email Format:</strong><br>
                    <span style="color: #00ffff; font-size: 0.9em;">username::quantum.foam</span>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="main-content">
                <!-- Inbox View -->
                <div id="inbox" class="content-section active">
                    <h2 class="section-header">Inbox</h2>
                    
                    <div class="success-message" id="inboxSuccess"></div>
                    <div class="error-message" id="inboxError"></div>
                    
                    <div id="emailList" class="email-list">
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <p>No emails yet</p>
                            <p style="margin-top: 10px; font-size: 0.9em;">Send yourself a test email to get started</p>
                        </div>
                    </div>
                    
                    <div id="emailDetail" style="display: none;">
                        <button class="btn-back" id="backToInbox">← Back to Inbox</button>
                        <div class="email-detail" id="emailDetailContent"></div>
                    </div>
                </div>
                
                <!-- Compose View -->
                <div id="compose" class="content-section">
                    <h2 class="section-header">Compose Email</h2>
                    
                    <div class="success-message" id="composeSuccess"></div>
                    <div class="error-message" id="composeError"></div>
                    
                    <form id="composeForm">
                        <div class="form-group">
                            <label for="fromInput">From:</label>
                            <input type="text" id="fromInput" readonly>
                        </div>
                        
                        <div class="form-group">
                            <label for="toInput">To: (username::quantum.foam)</label>
                            <input type="text" id="toInput" placeholder="recipient::quantum.foam" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="subjectInput">Subject:</label>
                            <input type="text" id="subjectInput" placeholder="Email subject" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="bodyInput">Message:</label>
                            <textarea id="bodyInput" placeholder="Type your message here..." required></textarea>
                        </div>
                        
                        <button type="submit" class="btn-primary">Send Email</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Get username from localStorage or use guest
        const username = localStorage.getItem('chat_username') || 'guest';
        const userEmail = `${username}::quantum.foam`;
        
        document.getElementById('currentUser').textContent = username;
        document.getElementById('userEmail').textContent = userEmail;
        document.getElementById('fromInput').value = userEmail;
        
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                
                // Update active button
                document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Show corresponding view
                document.querySelectorAll('.content-section').forEach(section => {
                    section.classList.remove('active');
                });
                document.getElementById(view).classList.add('active');
                
                // Load inbox if switching to it
                if (view === 'inbox') {
                    loadInbox();
                }
            });
        });
        
        // Load inbox
        async function loadInbox() {
            try {
                const response = await fetch(`/api/email/inbox/${username}`);
                const data = await response.json();
                
                const emailList = document.getElementById('emailList');
                
                if (data.inbox.length === 0) {
                    emailList.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <p>No emails yet</p>
                            <p style="margin-top: 10px; font-size: 0.9em;">Send yourself a test email to get started</p>
                        </div>
                    `;
                    return;
                }
                
                emailList.innerHTML = data.inbox.map(email => `
                    <div class="email-item ${email.read ? '' : 'unread'}" data-email-id="${email.id}">
                        <div class="email-header">
                            <span class="email-from">From: ${email.from}</span>
                            <span class="email-date">${new Date(email.timestamp).toLocaleString()}</span>
                        </div>
                        <div class="email-subject">${escapeHtml(email.subject)}</div>
                        <div class="email-preview">${escapeHtml(email.body.substring(0, 100))}${email.body.length > 100 ? '...' : ''}</div>
                    </div>
                `).join('');
                
                // Add click handlers
                document.querySelectorAll('.email-item').forEach(item => {
                    item.addEventListener('click', () => {
                        const emailId = item.dataset.emailId;
                        const email = data.inbox.find(e => e.id === emailId);
                        if (email) {
                            showEmailDetail(email);
                        }
                    });
                });
                
            } catch (error) {
                console.error('Failed to load inbox:', error);
                showError('inboxError', 'Failed to load inbox');
            }
        }
        
        // Show email detail
        function showEmailDetail(email) {
            const detailContent = document.getElementById('emailDetailContent');
            
            detailContent.innerHTML = `
                <div class="email-detail-header">
                    <div class="email-detail-subject">${escapeHtml(email.subject)}</div>
                    <div class="email-detail-meta"><strong>From:</strong> ${escapeHtml(email.from)}</div>
                    <div class="email-detail-meta"><strong>To:</strong> ${escapeHtml(email.to)}</div>
                    <div class="email-detail-meta"><strong>Date:</strong> ${new Date(email.timestamp).toLocaleString()}</div>
                </div>
                <div class="email-detail-body">${escapeHtml(email.body)}</div>
            `;
            
            document.getElementById('emailList').style.display = 'none';
            document.getElementById('emailDetail').style.display = 'block';
        }
        
        // Back to inbox
        document.getElementById('backToInbox').addEventListener('click', () => {
            document.getElementById('emailDetail').style.display = 'none';
            document.getElementById('emailList').style.display = 'flex';
        });
        
        // Send email
        document.getElementById('composeForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const from = document.getElementById('fromInput').value;
            const to = document.getElementById('toInput').value.trim();
            const subject = document.getElementById('subjectInput').value.trim();
            const body = document.getElementById('bodyInput').value.trim();
            
            // Validate email format
            if (!to.includes('::quantum.foam')) {
                showError('composeError', 'Invalid email format. Must be username::quantum.foam');
                return;
            }
            
            try {
                const response = await fetch('/api/email/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ from, to, subject, body })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to send email');
                }
                
                const data = await response.json();
                
                showSuccess('composeSuccess', 'Email sent successfully!');
                
                // Clear form
                document.getElementById('toInput').value = '';
                document.getElementById('subjectInput').value = '';
                document.getElementById('bodyInput').value = '';
                
                // Switch to inbox after 2 seconds
                setTimeout(() => {
                    document.querySelector('[data-view="inbox"]').click();
                }, 2000);
                
            } catch (error) {
                console.error('Failed to send email:', error);
                showError('composeError', 'Failed to send email: ' + error.message);
            }
        });
        
        // Helper functions
        function showSuccess(elementId, message) {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.classList.add('visible');
            setTimeout(() => {
                element.classList.remove('visible');
            }, 5000);
        }
        
        function showError(elementId, message) {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.classList.add('visible');
            setTimeout(() => {
                element.classList.remove('visible');
            }, 5000);
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Initial load
        loadInbox();
        
        // Auto-refresh inbox every 30 seconds
        setInterval(() => {
            if (document.getElementById('inbox').classList.contains('active')) {
                loadInbox();
            }
        }, 30000);
    </script>
</body>
</html>
