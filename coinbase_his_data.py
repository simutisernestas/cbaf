import json
import hmac
import hashlib
import time
import requests
import base64
from requests.auth import AuthBase
import time
import csv
import datetime

# Create custom authentication for Exchange


class CoinbaseExchangeAuth(AuthBase):
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

    def __call__(self, request):
        timestamp = str(time.time())
        message = timestamp + request.method + \
            request.path_url + (request.body or '')
        hmac_key = base64.b64decode(self.secret_key)
        signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest())

        request.headers.update({
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        })
        return request


api_url = 'https://api.pro.coinbase.com/'
fp = open('sand', 'r')
API_KEY = fp.readline().encode('utf-8').rstrip()
API_SECRET = fp.readline().encode('utf-8').rstrip()
API_PASS = fp.readline().encode('utf-8').rstrip()
auth = CoinbaseExchangeAuth(API_KEY, API_SECRET, API_PASS)

csvfile = open('eggs.csv', 'w', newline='')
spamwriter = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)

month = 60*60*24*365
back = 60*300
started = int(time.time())
end_stamp = int(time.time())
start_stamp = int(time.time())-back
while (started-end_stamp) < month:
    start = datetime.datetime.utcfromtimestamp(start_stamp).isoformat()
    end = datetime.datetime.utcfromtimestamp(end_stamp).isoformat()
    r = requests.get(
        api_url + f'products/BTC-EUR/candles?granularity=60&start={start}&end={end}', auth=auth)
    if not isinstance(r.json(), list):
        print(r.json())
        continue
    for p in r.json():
        spamwriter.writerow(p)
    end_stamp = start_stamp
    start_stamp -= back
    time.sleep(0.33333)