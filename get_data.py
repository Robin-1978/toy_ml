import requests
from bs4 import BeautifulSoup
import pandas as pd


def get_ssq_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    data = []
    for table in soup.find_all('table'):
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) > 0:
                data.append([cell.text.strip() for cell in cells])

    return data

data = get_ssq_data('https://cjcp.cn/kaijiang/ssq/')
#save as csv
df = pd.DataFrame(data)
df.to_csv('data/ssq/data1.csv', index=False)