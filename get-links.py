from requests.auth import HTTPBasicAuth
import requests
import json

with open("api_key.txt", "r") as key:
     api_key = key.read()

auth = HTTPBasicAuth('apikey', api_key.strip())
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

tags = ["environment" , "politics", "football", "food"]


for tag in tags:
    for year in range(0, 9):
        for month in range(1, 13):
            from_date = f"20{year:02}-{month:02}-01" # yyyy-mm-dd
            to_date = f"201{year}-{month}-30"
            api_key = "63581936-8fd7-40f7-ac80-76cb937c59a9"
            url = f"https://content.guardianapis.com/search?tag={tag}/{tag}&api-key={api_key}&from-date={from_date}&to-date={to_date}&page-size=50"
            response = requests.get(url, headers=headers, auth=auth)
            with open(f"./responses/{month}-201{year}-{tag}.json", "w") as f:
                    json.dump(response.json(), f)

