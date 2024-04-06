import requests

url = 'http://localhost:9200/_nodes?pretty'
response = requests.get(url)

if response.status_code == 200:
    print(response.json())
else:
    print("Error:", response.status_code)
