from bs4 import BeautifulSoup
from requests.auth import HTTPBasicAuth
import requests

auth = HTTPBasicAuth('apikey', '63581936-8fd7-40f7-ac80-76cb937c59a9')
url = 'https://content.guardianapis.com/search?tag=politics/politics&api-key=63581936-8fd7-40f7-ac80-76cb937c59a9&page-size=50'
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
response = requests.get(url, headers=headers, auth=auth)

print(response.text)



# "sectionName":"Environment"
# "webTitle":"Lords to debate mandating swift bricks in new homes in England"
# "webUrl":"https://www.theguardian.com/environment/2023/sep/04/lords-to-debate-mandating-swift-bricks-in-new-uk-homes"



# # data is used to store the text of the HTML document of the site
# data = response.text
# # soup parses that text file
# soup = BeautifulSoup(data, "json.parser")

# print(soup.prettify())

# # get news article links
# for link in soup.find_all("a", {"class":"link d-block p-relative card__image-container"}):
#     print(link.get("href"))
