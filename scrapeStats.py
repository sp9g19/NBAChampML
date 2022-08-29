from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from webdriver_manager.chrome import ChromeDriverManager

team_codes = ["ATL", "BOS", "NJN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
              "MIA", "MIL", "MIN", "NOH", "NYK", "SEA", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]


def check_and_update_code(code, year_in):
    if code == "NJN" and int(year_in) > 2012:
        return "BRK"
    if code == "CHA":
        if int(year_in) < 2005:
            return "NO CHA"
        if int(year_in) > 2014:
            return "CHO"
    if code == "NO CHA" and int(year_in) > 2004:
        return "CHA"
    if code == "NOH":
        if int(year_in) < 2006:
            return "NOH"
        if 2005 < int(year_in) < 2008:
            return "NOK"
        if 2013 < int(year_in):
            return "NOP"
    if code == "NOK" and 2007 < int(year_in):
        return "NOH"
    if code == "SEA" and int(year_in) > 2008:
        return "OKC"
    return code


if __name__ == "__main__":
    driver = webdriver.Chrome(ChromeDriverManager().install())
    for team in team_codes:
        for y in range(3, 23):
            year = str(2000 + y)
            team = check_and_update_code(team, year)
            if team == "NO CHA":
                continue
            url = "https://www.basketball-reference.com/teams/" + team + "/" + year + ".html"
            driver.get(url)
            c = driver.page_source
            soup = BeautifulSoup(c, "lxml")
            table = soup.find('table', id="team_and_opponent")
            df = pd.read_html(table.prettify())[0]
            Path("C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\"+team+"\\"+year).mkdir(parents=True, exist_ok=True)
            df.to_csv("C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\" + team + "\\" + year + "\\" + team + year + ".csv")

    for y in range(3, 23):
        year = str(2000 + y)
        url = "https://www.basketball-reference.com/playoffs/NBA_" + year + ".html"
        driver.get(url)
        c = driver.page_source
        soup = BeautifulSoup(c, "lxml")
        table = soup.find('table', id="advanced-team")
        df = pd.read_html(table.prettify())[0]
        df.to_csv("C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\playoffs\\" + year + ".csv")
