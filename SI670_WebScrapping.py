import pandas as pd
import requests
from bs4 import BeautifulSoup

# Created empty dataframe
dataset = pd.DataFrame(columns=['Title', 'Location', 'Type', 'Date / Time', 'Further Comments'])

# Base URL - England - South East
BASE_URL = "https://www.paranormaldatabase.com/regions/southeast.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - South West - England
BASE_URL = "https://www.paranormaldatabase.com/regions/southwest.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Greater London - England
BASE_URL = "https://www.paranormaldatabase.com/regions/greaterlondon.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
urls_to_scrape[9] = "/reports/underground.php"
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1


# Base URL - East of England - England
BASE_URL = "https://www.paranormaldatabase.com/regions/eastengland.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - East Midlands - England
BASE_URL = "https://www.paranormaldatabase.com/regions/eastmidlands.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - West Midlands - England
BASE_URL = "https://www.paranormaldatabase.com/regions/westmidlands.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - The North East & Yorkshire - England
BASE_URL = "https://www.paranormaldatabase.com/regions/northeastandyorks.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Scotland
BASE_URL = "https://www.paranormaldatabase.com/regions/scotland.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Wales
BASE_URL = "https://www.paranormaldatabase.com/regions/wales.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Northern Ireland
BASE_URL = "https://www.paranormaldatabase.com/regions/northernireland.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Replublic of Ireland
BASE_URL = "https://www.paranormaldatabase.com/regions/ireland.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Other Regions
BASE_URL = "https://www.paranormaldatabase.com/regions/otherregions.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Browse Places
BASE_URL = "https://www.paranormaldatabase.com/reports/reports.htm"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Browse Type
BASE_URL = "https://www.paranormaldatabase.com/reports/reports-type.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
urls_to_scrape[1] = "/reports/animal.php"
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Browse People
BASE_URL = "https://www.paranormaldatabase.com/reports/reports-people.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
urls_to_scrape[0] = "/reports/anneboleyn.php"
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1


# Base URL - Paranomal Calendar
BASE_URL = "https://www.paranormaldatabase.com/calendar/Pages/calendar.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")


all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1

# Base URL - Recent Additions
BASE_URL = "https://www.paranormaldatabase.com/recent/index.php"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")
pnum = []

for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if "pageNum" in url:
        urls = url.split("&")
        num_string = ''.join(filter(str.isdigit, urls[0]))
        pnum.append(int(num_string))

if len(pnum) == 0:
  pnum.append(0)
  pnum.append(0)

print(pnum)
# Initialize page counter
page_number = 0

while page_number <= pnum[-1]:
    # Construct the URL for the current page
    if page_number == 0:
        url = BASE_URL
    else:
        #url = f"{BASE_URL}?page={page_number}"
        url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

    # Fetch the page
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve the page.")
        break

    # Parse the page with Beautiful Soup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the story blocks
    story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
    if not story_blocks:  # Stop if no more stories are found
        print("No more stories found. Exiting.")
        break

    # Loop through each story block
    for story in story_blocks:
        # Extract the title
        h4_tag = story.find('h4')
        title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

        if title == "N/A":
          continue

        # Extract details from the <p> tag
        details = story.find_all('p')[-1]
        details_html = details.decode_contents() if details else ""
        lines = details_html.split('<br/>')

        # Initialize fields
        location = type_text = date_text = comments = "N/A"

        # Extract each detail line by line
        for line in lines:
            line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
            if line.startswith("Location:"):
                location = line.replace("Location:", "").strip()
            elif line.startswith("Type:"):
                type_text = line.replace("Type:", "").strip()
            elif line.startswith("Date / Time:"):
                date_text = line.replace("Date / Time:", "").strip()
            elif line.startswith("Further Comments:"):
                comments = line.replace("Further Comments:", "").strip()

        dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

    # Move to the next page
    print(f"Finished scraping page {page_number}. Moving to the next page...")
    page_number += 1

# Base URL - North West - England
BASE_URL = "https://www.paranormaldatabase.com/regions/northwest.html"
page = requests.get(BASE_URL)
soup = BeautifulSoup(page.content, "html.parser")

all_urls = []
for a_tag in soup.find_all('a'):
    url = a_tag.get('href')
    if url:
      all_urls.append(url)

all_urls

start_index = all_urls.index("/index.html", 1)
urls_to_scrape = all_urls[start_index+1:len(all_urls)-1]
print(len(urls_to_scrape))
urls_to_scrape

for i in urls_to_scrape:
  BASE_URL = "https://www.paranormaldatabase.com" + i
  page = requests.get(BASE_URL)
  soup = BeautifulSoup(page.content, "html.parser")
  pnum = []

  for a_tag in soup.find_all('a'):
      url = a_tag.get('href')
      if "pageNum" in url:
          urls = url.split("&")
          num_string = ''.join(filter(str.isdigit, urls[0]))
          pnum.append(int(num_string))

  if len(pnum) == 0:
    pnum.append(0)
    pnum.append(0)

  print(pnum)
  # Initialize page counter
  page_number = 0

  while page_number <= pnum[-1]:
      # Construct the URL for the current page
      if page_number == 0:
          url = BASE_URL
      else:
          #url = f"{BASE_URL}?page={page_number}"
          url = f"{BASE_URL}?pageNum_paradata={page_number}&totalRows_paradata=92"

      # Fetch the page
      response = requests.get(url)
      if response.status_code != 200:
          print("Failed to retrieve the page.")
          break

      # Parse the page with Beautiful Soup
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find all the story blocks
      story_blocks = soup.find_all('div', class_="w3-border-left w3-border-top w3-left-align")
      if not story_blocks:  # Stop if no more stories are found
          print("No more stories found. Exiting.")
          break

      # Loop through each story block
      for story in story_blocks:
          # Extract the title
          h4_tag = story.find('h4')
          title = h4_tag.find('span').get_text(strip=True) if h4_tag and h4_tag.find('span') else "N/A"

          if title == "N/A":
            continue

          # Extract details from the <p> tag
          details = story.find_all('p')[-1]
          details_html = details.decode_contents() if details else ""
          lines = details_html.split('<br/>')

          # Initialize fields
          location = type_text = date_text = comments = "N/A"

          # Extract each detail line by line
          for line in lines:
              line = BeautifulSoup(line, 'html.parser').get_text(strip=True)
              if line.startswith("Location:"):
                  location = line.replace("Location:", "").strip()
              elif line.startswith("Type:"):
                  type_text = line.replace("Type:", "").strip()
              elif line.startswith("Date / Time:"):
                  date_text = line.replace("Date / Time:", "").strip()
              elif line.startswith("Further Comments:"):
                  comments = line.replace("Further Comments:", "").strip()

          dataset = pd.concat([dataset, pd.DataFrame([{'Title': title, 'Location': location, 'Type': type_text, 'Date / Time': date_text, 'Further Comments': comments}])], ignore_index=True)

      # Move to the next page
      print(f"Finished scraping page {page_number}. Moving to the next page...")
      page_number += 1