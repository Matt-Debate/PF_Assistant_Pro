import time
import logging

# Configure logging (you can customize this as needed)
logging.basicConfig(
    filename='app_api_calls.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def log_and_retry(func, max_retries=3, backoff_factor=2, initial_delay=1):
    """
    Calls the passed function with retry logic and logs each attempt.

    Args:
        func: A no-argument callable that performs the API call.
        max_retries: Maximum number of retries before giving up.
        backoff_factor: Multiplier for delay between retries.
        initial_delay: Delay before first retry in seconds.

    Returns:
        The result of func() if successful.

    Raises:
        Exception from the last failed attempt after max retries.
    """
    delay = initial_delay
    attempt = 0

    while attempt <= max_retries:
        try:
            logging.info(f"Attempt {attempt + 1} of {max_retries + 1} for function {func}")
            result = func()
            logging.info(f"Success on attempt {attempt + 1}")
            return result
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt > max_retries:
                logging.error(f"Max retries exceeded. Raising exception.")
                raise
            logging.info(f"Retrying after {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor
