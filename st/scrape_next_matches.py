from playwright.sync_api import sync_playwright
import csv
from datetime import datetime, timedelta


def format_date(date_str):
    today = datetime.today()
    if "Today" in date_str:
        return today.strftime("%Y-%m-%d")
    elif "Yesterday" in date_str:
        yesterday = today - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")
    elif "Tomorrow" in date_str:
        tomorrow = today + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")
    else:
        date_obj = datetime.strptime(date_str[:11], "%d %b %Y")
        return date_obj.strftime("%Y-%m-%d")


def scrape():
    with sync_playwright() as p:
        leagues = [
            {"type": "football", "region": "england", "league": "premier-league"},
            {"type": "football", "region": "usa", "league": "mls"},
            {"type": "basketball", "region": "usa", "league": "nba"},
            {"type": "american-football", "region": "usa", "league": "nfl"},
            # {"type": "baseball", "region": "usa", "league": "mlb"},
        ]
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_viewport_size({"width": 1800, "height": 10000})
        end_date = datetime.today() + timedelta(days=7)
        for league_info in leagues:
            league_type = league_info["type"]
            region = league_info["region"]
            league = league_info["league"]
            total = []
            try:
                page.goto(f"https://www.oddsportal.com/{league_type}/{region}/{league}")
                page.wait_for_selector("div.eventRow")
                events = page.query_selector_all("div.eventRow")

                date_str = ""
                for event in events:
                    children = event.query_selector_all(":scope > *")
                    if len(children) > 1:
                        date_str = format_date(
                            children[-2]
                            .query_selector("div:nth-child(1) div:nth-child(1)")
                            .text_content()
                        )
                    if datetime.strptime(date_str, "%Y-%m-%d") > end_date:
                        break
                    p_element = children[-1].query_selector_all("p.participant-name")
                    odds_element = children[-1].query_selector_all(
                        "p.default-odds-bg-bgcolor"
                    )
                    goals_elements = children[-1].query_selector_all(
                        "div:nth-child(1) a:nth-child(1) div:nth-child(1) > div:nth-child(2) div:nth-child(1) div:nth-child(1) > div:nth-child(2) div:nth-child(1) div"
                    )
                    if len(goals_elements) > 0:
                        continue
                    if len(odds_element) == 0:
                        odds_element = children[-1].query_selector_all(
                            "span.default-odds-bg-bgcolor p"
                        )
                    total.append(
                        [date_str]
                        + [
                            names.text_content().split("(")[0].strip()
                            for names in p_element
                        ]
                        + [odds.text_content() for odds in odds_element]
                    )
            except Exception as err:
                print(f"Error on {league}")

            file_path = f"./data/next_matches/{region}-{league}.csv"
            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                header = [
                    "Date",
                    "HomeTeam",
                    "AwayTeam",
                    "ODDS1",
                ]
                if league_type == "football":
                    header.append("ODDSX")
                header.append("ODDS2")
                writer.writerow(header)
                writer.writerows(total)
        print("Done!")
        browser.close()


if __name__ == "__main__":
    scrape()
