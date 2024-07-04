import requests 
import numpy as np
from bs4 import BeautifulSoup


def download_NSIDC(url, year_start, year_end):
    
    """
        This is the function used to download the nsidc daily netcdf datafiles. 
        
        You need to specify the url from the NSIDC website that is hosting the data. 
        The CDR dataset is stored on 
            url = "https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/daily/"
        
        And the CDR-NRT is stored on
            url = "https://noaadata.apps.nsidc.org/NOAA/G10016_V2/north/daily/"
        
    """
    years = np.arange(year_start, year_end)
    
    print(years)
    for year in years : 
        str_year = str(year)
        archive_url = url+str_year+'/'
        print(archive_url)
        r = requests.get(archive_url)
        data = BeautifulSoup(r.text, "html.parser")

        for l in data.find_all("a")[1:]:	
            r = requests.get(archive_url + l["href"])	
            with open(l["href"], "wb") as f:    	         
                f.write(r.content)
                
download_NSIDC("https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/daily/", 2023, 2024)