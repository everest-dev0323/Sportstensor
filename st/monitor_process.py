import schedule
import time
import subprocess
import os
import logging

# Set up logging configuration
log_filename = "./logging/process_log.log"
os.makedirs("./logging", exist_ok=True)
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Define the function to run a specific script
def run_script(script_name):
    try:
        # Path to the virtual environment Python interpreter
        venv_python = os.path.join("..", ".venv", "bin", "python")

        # Run the specified script
        script_path = os.path.join(script_name)

        # Log start time
        logging.info(f"Starting - {script_name}...")

        result = subprocess.run(
            [venv_python, script_path], capture_output=True, text=True
        )
        
        logging.info(result.stdout.replace('\n', '  @:@  '))

        # Log end time and output
        logging.info(f"Completed - {script_name}")
        if result.stderr:
            logging.error(f"Errors:\n{result.stderr}")
    except Exception as e:
        logging.error(f"Error running {script_name}: {e}")


# Define a function to run the scripts in sequence
def run_scripts_sequentially():
    # Run predictor.py first
    run_script("./scrape_next_matches.py")
    logging.info(f"\n")

# Schedule the sequential execution of scripts every 3 days
run_scripts_sequentially()
print("Scraping process monitor is running now!")
schedule.every(30).minutes.do(run_scripts_sequentially)

# Keep the script running to check the schedule
while True:
    schedule.run_pending()
    time.sleep(1)  # Check every second
