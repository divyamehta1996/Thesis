from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
import re
import requests
import urllib2
import pandas as pd
import numpy as np
import sys

reload(sys)
sys.setdefaultencoding('utf8')

def write_file(data, labels):
	df = pd.DataFrame(np.array(data))
	lf = pd.DataFrame(np.array(labels))
	nf = pd.DataFrame(np.array(neutral))
	df.to_csv('allSidesData.csv', index=False)
	lf.to_csv('allSidesLabels.csv', index=False)
	nf.to_csv('allSidesNeutral.csv', index=False)
	print len(df)
	print len(lf)
	print len(nf)

def retrieve():
	resp  = requests.get("https://www.allsides.com/story-list")
	# # print r.content
	# soup = BeautifulSoup(r.content, 'html.parser')
	# # print(soup.prettify())
	# letters = soup.find_all("div", class_="feature-thumbs span4")

	# for letter in letters:
	# 	print letter.find_all("div", class_="news-title")[0].a.get_text()
	# 	print letter.find_all("div", class_="global-bias")[0].get_text()

	http_encoding = resp.encoding if 'charset' in resp.headers.get('content-type', '').lower() else None
	html_encoding = EncodingDetector.find_declared_encoding(resp.content, is_html=True)
	encoding = html_encoding or http_encoding
	soup = BeautifulSoup(resp.content, from_encoding=encoding)
	data = []
	labels = []
	# neutral = []
	i = 0
	for link in soup.find_all('a', href=True):
		if link['href'].startswith("/story/"):
			print link['href']
			r  = requests.get("https://www.allsides.com/" + link['href'])
			soup = BeautifulSoup(r.content, 'html.parser')
			# print(soup.prettify())
			letters = soup.find_all("div", class_="quicktabs-views-group")
			for letter in letters:
				headline = letter.find_all("div", class_="news-title")[0].a.get_text()
				headline.encode('utf-8')
				bias = letter.find_all("div", class_="global-bias")[0].get_text()
				if "Left" in bias:
					data.append(headline)
					labels.append(1)
					# print data[i], labels[i]
					i = i + 1
				elif "Right" in bias:
					data.append(headline)
					labels.append(0)	
					# print data[i], labels[i]
					i = i + 1
	 			elif "Center" in bias:
	 				neu = 0
	 				# neutral.append(headline)
	 			else:
					print bias

	return data, labels


def main():
	data, labels = retrieve()
	# np.array(data)
	# np.array(labels)
	# print data


if __name__ == "__main__":
    main()

