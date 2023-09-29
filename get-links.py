from bs4 import BeautifulSoup
import requests

url = "https://www.bbc.co.uk/news"
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
response = requests.get(url, headers=headers)

# data is used to store the text of the HTML document of the site
data = response.text
# soup parses that text file
soup = BeautifulSoup(data, "html.parser")

print(soup.prettify())

# get news article links
# for link in soup.find_all("a", {"class":"link d-block p-relative card__image-container"}):
#     print(link.get("href"))
