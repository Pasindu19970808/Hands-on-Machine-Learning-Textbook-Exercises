import json
import urllib.request
import urllib.parse
import ssl
#using API to get data in json format and parsing it 
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://api.currencyscoop.com/v1/latest?"

parameters = dict()
parameters["base"] = "USD"
parameters["api_key"] = "1f406095ad9a745753d6ce98d6fc175a"

final_url = url + urllib.parse.urlencode(parameters, safe = ",")

#creates only a request object with a browser header
request = urllib.request.Request(final_url, headers={'User-Agent':'Mozilla/5.0'})
total_data = json.loads(urllib.request.urlopen(request).read().decode('utf-8'))

rate_data  = total_data["response"]["rates"]
#making a ordered list of rates
sorted_rates = {val[0] : val[1] for val in sorted(rate_data.items(), key=lambda x : x[1], reverse=True)}
print(sorted_rates)