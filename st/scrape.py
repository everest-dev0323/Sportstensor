from playwright.sync_api import sync_playwright
import os
import csv
from datetime import datetime, timedelta

start_year = 2024
end_year = 2024
leagues = [
    # {"type": "football", "region": "england", "league": "premier-league"},
    {"type": "football", "region": "usa", "league": "mls", "season_type": 1},
    # {"type": "basketball", "region": "usa", "league": "nba"},
    # {"type": "american-football", "region": "usa", "league": "nfl"},
    # {"type": "baseball", "region": "usa", "league": "mlb", "season_type": 1},
]


def format_date(date_str):
    today = datetime.today()
    if "Today" in date_str:
        return today.strftime("%Y-%m-%d")
    elif "Yesterday" in date_str:
        yesterday = today - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")
    else:
        date_obj = datetime.strptime(date_str[:11], "%d %b %Y")
        return date_obj.strftime("%Y-%m-%d")


def scrape():
    failed = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1800, "height": 50000})
        for league_info in leagues:
            league_type = league_info["type"]
            region = league_info["region"]
            league = league_info["league"]
            season_type = league_info.get("season_type", None)

            for year in range(start_year, end_year + 1):
                url = f"https://www.oddsportal.com/{league_type}/{region}/{league}{'-'+str(year) if season_type and year != 2024 else ''}{f'-{year}-{year + 1}' if year != 2024 and not season_type else ''}/results/#/page/1/"
                print(f"Scraping {url}")
                try:
                    page.goto(url)
                    total = []
                    count = 0
                    while True:
                        # page.wait_for_timeout(20000)
                        print(count)
                        page.wait_for_load_state("domcontentloaded")
                        page.wait_for_selector("div.eventRow")
                        events = page.query_selector_all("div.eventRow")
                        date_str = ""
                        for event in events:
                            try:
                                children = event.query_selector_all(":scope > *")
                                if len(children) > 1:
                                    date_str = format_date(
                                        children[-2]
                                        .query_selector(
                                            "div:nth-child(1) div:nth-child(1)"
                                        )
                                        .text_content()
                                        if children[-2]
                                        else ""
                                    )

                                p_element = children[-1].query_selector_all(
                                    "p.participant-name"
                                )
                                odds_element = children[-1].query_selector_all(
                                    "p.default-odds-bg-bgcolor"
                                )
                                goals_elements = children[-1].query_selector_all(
                                    "div:nth-child(1) a:nth-child(1) div:nth-child(1) > div:nth-child(2) div:nth-child(1) div:nth-child(1) > div:nth-child(2) div:nth-child(1) div"
                                )
                                if len(goals_elements) == 0 or len(odds_element) == 0:
                                    continue
                                goal_home = int(goals_elements[0].text_content())
                                goal_away = int(goals_elements[1].text_content())
                                result = (
                                    "D"
                                    if goal_home == goal_away
                                    else "H" if goal_home > goal_away else "A"
                                )
                                total.append(
                                    [date_str, year]
                                    + [
                                        names.text_content().split("(")[0].strip()
                                        for names in p_element
                                    ]
                                    + [
                                        goal_home,
                                        goal_away,
                                        result,
                                    ]
                                    + [odds.text_content() for odds in odds_element]
                                )

                            except Exception as e:
                                print(f"Error parsing event: {e}")

                        next_button = page.query_selector(
                            "a.pagination-link:last-child"
                        )
                        if next_button and next_button.text_content() == "Next":
                            next_button.click()
                            count += 1
                            page.wait_for_timeout(100)
                        else:
                            break

                    # Save results to CSV
                    directory = f"data/{region}-{league}"
                    os.makedirs(directory, exist_ok=True)
                    file_path = f"{directory}/{year}-{year + 1}.csv"
                    with open(
                        file_path, mode="w", newline="", encoding="utf-8"
                    ) as file:
                        writer = csv.writer(file)
                        header = [
                            "Date",
                            "Season",
                            "HomeTeam",
                            "AwayTeam",
                            "FTHG",
                            "FTAG",
                            "FTR",
                            "ODDS1",
                        ]
                        if league_type == "football":
                            header.append("ODDSX")
                        header.append("ODDS2")
                        writer.writerow(header)
                        writer.writerows(total[::-1])

                except Exception as err:
                    print(f"🧨 Error scraping {url}: {err}")
                    failed.append(url)

        print("Done!")
        print("Failed scrapes:", failed)
        browser.close()


if __name__ == "__main__":
    scrape()
