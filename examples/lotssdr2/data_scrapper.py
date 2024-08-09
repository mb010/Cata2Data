import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
from urllib.parse import urljoin
import argparse
from tqdm import tqdm

# Function to get the full-resolution links from a single page
def get_links(page_url, download_link='^public/DR2/mosaics/.*/mosaic-blanked.fits$'):
    response = requests.get(page_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    hrefs = soup.find_all('a', href=True)
    data_links = []
    for ref in hrefs:
        if re.search(download_link, ref['href']):
            data_links.append(ref['href'])
    
    return data_links

def download_all_files(links, base_url, test=False, dir="./data"):
    for link in tqdm(links):
        try:
            download_file(base_url, link, local_dir=dir)
        except Exception as e:
            print(f"Error downloading {urljoin(base_url+link)}: {e}")
        if test:
            break

def download_file(url, file_name, local_dir='./data', chunk_size=8192):
    full_url = urljoin(url, file_name)

    local_dir = Path(local_dir)
    local_filename = local_dir / Path(file_name)
    local_filename.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(full_url, stream=True) as r:
        r.raise_for_status()
        with local_filename.open('wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # Skip empty chunks
                    f.write(chunk)
                    
    return str(local_filename)  # Return the file path as a string


def main(url, base_url, download_link, test=False, dir='./data'):
    links = get_links(url, download_link)
    download_all_files(links, base_url, test, dir=dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download data from the LOFAR surveys website")
    parser.add_argument('--url', help="The URL of the page to scrape", default='https://lofar-surveys.org/dr2_release.html')
    parser.add_argument('--base_url', help="The base URL of the website", default='https://lofar-surveys.org/')
    parser.add_argument('--download_link', help="The regex pattern to match the download link", default='^public/DR2/mosaics/.*/mosaic-blanked.fits$')
    parser.add_argument('--test', action='store_true', help="Only try to download the first file")
    parser.add_argument('--dir', help="The local directory in which to save the data.", default='./data')
    args = parser.parse_args()

    main(args.url, args.base_url, args.download_link, args.test, args.dir)
