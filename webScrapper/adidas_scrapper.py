import os.path
import requests
from bs4 import BeautifulSoup as bs
from requests_file import FileAdapter

def get_text(url):
    s = requests.Session()
    s.mount('file://', FileAdapter())

    resp = s.get(url)

    page_soup = bs(resp.text, 'html.parser')

    #blueprint how to access data
    """#GET author
    results = page_soup.find_all('html_tag', class_='gl-label--bold')
    if len(results) > 0:
        author = [result.get_text() for result in results]"""

    #GET description
    results = page_soup.find_all('div', class_='gl-vspacing-m review_text___JBgtJ')
    if len(results) > 0:
        description = [result.get_text() for result in results]

    #error correction of comment data
    description = list(map(lambda st: str.replace(st, "\n                                    ", " "), description))
    description = list(map(lambda st: str.replace(st, "\n                                ", " "), description))
    description = list(map(lambda st: str.replace(st, "\n", " "), description))
    
    return description

if __name__ == '__main__':
    #acces local html site
    url = 'file:///home/ji78remu/Schreibtisch/CaseStudySeminar/webScrapper/htmlPages/con80_white.html'

    #get raw_comment_data
    text = get_text(url)

    #save to local directory
    save_path = '/home/ji78remu/Schreibtisch/CaseStudySeminar/txtSaver'
    fileName = 'con80_white'
    completeName = os.path.join(save_path, fileName+'.txt')

    #create local file
    f = open(completeName, "w+")
    for comment in text:
        f.write(comment+"\n")
    f.close()

    print('successfully scrapped: ', len(text), ' comments for '+fileName)
