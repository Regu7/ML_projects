# IMPORT ALL NECESSARY LIBRARIES
import csv

import requests
from bs4 import BeautifulSoup

# LINK TO THE WEB PAGE FOR SCRAPPYING
URL = "https://www.canadavisa.com/express-entry-invitations-to-apply-issued.html"

# GET THE WEB PAGE
page = requests.get(URL)

# CREATE A SOUP
soup = BeautifulSoup(page.content, "html5lib")

# FIND ALL TABLES
TABLE = soup.find_all("table", class_="table table-bordered")
print(f"TOTAL NO TABLES: {len(TABLE)}")

# STORE WEB PAGE IN HTML FOR ANALYSIS
with open("web_PAGE.html", "wb") as writer:
    # Alternatively you could use
    writer.write(soup.encode("utf-8"))

# CREATE A CSV WRITER
csv_writer = csv.writer(open("data/raw/CRS_Data.csv", "w", newline=""))

# WRITE ALL DATA IN CSV FILE
for table in TABLE:
    for row in table.find_all("tr"):
        headers = row.find_all("th")  # GET THE HEADER OF EACH TABLE
        columns = row.find_all("td")  # GET THE DATA OF EACH TABLE
        CONTENT = [head.get_text() for head in headers] + [
            column.get_text() for column in columns
        ]  # ADD HEADER AND DATA
        csv_writer.writerow(CONTENT)  # WRITE THE CONTENT IN CSV FILE
