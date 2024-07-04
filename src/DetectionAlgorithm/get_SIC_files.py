import requests 
import numpy as np
from bs4 import BeautifulSoup

years = np.arange(2019, 2024)
url = "https://data.seaice.uni-bremen.de/modis_amsr2/netcdf/Arctic/"
# url = "https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/daily/"

for year in years : 
# year = 2023
    str_year = str(year)
    archive_url = url+str_year+'/'
    print(archive_url)
    r = requests.get(archive_url)
    data = BeautifulSoup(r.text, "html.parser")
    # print(r)
    
    # print(data.find_all("a")[3:])
    for l in data.find_all("a")[5:]:	
        print(l)
        r = requests.get(archive_url + l["href"])	
        with open(l["href"], "wb") as f:    	         
            f.write(r.content)
