from scrapeStats import check_and_update_code
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
from webdriver_manager.chrome import ChromeDriverManager

team_codes = ["ATL", "BOS", "NJN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
              "MIA", "MIL", "MIN", "NOH", "NYK", "SEA", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]

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
            table = soup.find('table', id="team_misc")
            df = pd.read_html(table.prettify())[0]
            Path("C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\"+team+"\\"+year).mkdir(parents=True, exist_ok=True)
            df.to_csv("C:\\Users\\Sol Parker\\OneDrive\\Documents\\NBAChampML\\dataframes\\" + team + "\\" + year + "\\" + team + year + "Misc.csv")