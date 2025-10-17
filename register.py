<!DOCTYPE html>
<html>
<head>
    <title>Register</title>
    <style> /* Same as login */ </style>
</head>
<body>
    <h1>Register for Quantum Entanglement</h1>
    <form action="/register" method="post">
        <input type="text" name="username" placeholder="Username (for @hackah.edu)" required>
        <input type="password" name="password" placeholder="Password" required>
        <input type="text" name="labels" placeholder="Labels (for matching, comma-separated)">
        <button type="submit">Register & Get Email</button>
    </form>
    <p>Your email will be {username}@hackah.edu</p>
</body>
</html>
