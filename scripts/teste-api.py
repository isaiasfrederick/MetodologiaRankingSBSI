import json
import requests

import json

from sys import argv


app_id  = u'e405c4bd'
app_key  = u'5e3958b4a87bd9f46471e1963ab3b2c1'
language_code = "en"

word_id = argv[1]
endpoint = "entries"

# entries

url = "https://od-api.oxforddictionaries.com:443/api/v2/" + endpoint + "/" + language_code + "/" + word_id.lower()

print("\n")
raw_input(url)
print("\n")

r = requests.get(url, headers = {"app_id": app_id, "app_key": app_key})

print("\n")
print("code {}\n".format(r.status_code))
print("\n")
print("text \n" + r.text)
print("\n")
print("json \n" + json.dumps(r.json(), indent=4))
print("\n\n")
