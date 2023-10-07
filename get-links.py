from requests.auth import HTTPBasicAuth
import requests
import json

auth = HTTPBasicAuth('apikey', '63581936-8fd7-40f7-ac80-76cb937c59a9')
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

tags = ["environment" , "politics", "technology", "science", "society", "football", "food"]

for tag in tags:
    i = 10
    from_date = "2022-12-01" # yyyy-mm-dd
    to_date = "2022-12-31"
    api_key = "63581936-8fd7-40f7-ac80-76cb937c59a9"
    url = f"https://content.guardianapis.com/search?tag={tag}/{tag}&api-key={api_key}&from-date={from_date}&to-date={to_date}&page-size=50"
    response = requests.get(url, headers=headers, auth=auth)
    with open(f"./responses/{tag}{i}.json", "w") as f:
        json.dump(response.json(), f)

        