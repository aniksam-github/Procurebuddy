import urllib.request, json, time, subprocess

def request(url, method='GET', data=None, headers=None):
    if headers is None: headers = {}
    if data:
        data = json.dumps(data).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        resp = urllib.request.urlopen(req)
        return resp.getcode(), resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

email = 'test888@gmail.com'
print('Register Start:', request('http://localhost:8080/api/auth/register/start', 'POST', {'email': email, 'password': 'Password123!', 'name': 'Test 888'}))

time.sleep(1)
result = subprocess.run(['docker', 'exec', 'procurebuddy-db', 'psql', '-U', 'procurebuddy', '-d', 'procurebuddy', '-t', '-c', f"SELECT otp FROM pending_otps WHERE email = '{email}'"], capture_output=True, text=True)
otp = result.stdout.strip()
print('OTP:', otp)

status, body = request('http://localhost:8080/api/auth/register/verify', 'POST', {'email': email, 'otp': otp, 'password': 'Password123!'})
print('Verify:', status, body[:100])

if status == 200:
    token = json.loads(body)['token']
    print('Chats:', request('http://localhost:8080/api/chats', 'GET', headers={'Authorization': 'Bearer ' + token}))
    print('Message:', request('http://localhost:8080/api/chats/new-chat-id/message', 'POST', {'message': 'Hello'}, headers={'Authorization': 'Bearer ' + token}))
