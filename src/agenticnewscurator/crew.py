from crewai import Agent, Crew, Process, Task
import logging
import time
from functools import wraps

from crewai.project import CrewBase, after_kickoff, agent, before_kickoff, crew, task
from litellm.exceptions import RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from litellm.exceptions import RateLimitError

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom Agent class with rate limit retry handling
class RateLimitRetryAgent(Agent):
    def execute_task(self, task, context=None, tools=None):
        max_retries = 5
        base_wait = 5
        attempt = 0
        
        while attempt < max_retries:
            try:
                return super().execute_task(task, context, tools)
            except RateLimitError:
                attempt += 1
                wait_time = base_wait * (2**attempt)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time}s (Attempt {attempt}/{max_retries})")
                if attempt >= max_retries:
                    logging.error("Max retries reached. Task failed due to rate limits.")
                    raise
                time.sleep(wait_time)

# Add retry capability to LiteLLM completion
def retry_litellm_call(max_retries=5, base_wait=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    attempt += 1
                    wait_time = base_wait * (2**attempt)  # Exponential backoff
                    logging.warning(f"Rate limit hit. Retrying in {wait_time}s (Attempt {attempt}/{max_retries})")
                    if attempt >= max_retries:
                        logging.error("Max retries reached. LiteLLM call failed.")
                        raise e
                    time.sleep(wait_time)
        return wrapper
    return decorator

# Apply the retry decorator to litellm.completion
try:
    import litellm
    original_completion = litellm.completion
    litellm.completion = retry_litellm_call()(original_completion)
    logging.info("Successfully patched litellm.completion with retry capability")
except ImportError:
    logging.warning("litellm not available. Rate limit retry for LiteLLM not applied.")


@CrewBase
class Agenticnewscurator:
    """Agenticnewscurator crew"""
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    @before_kickoff
    def pull_data_example(self, inputs):
        # Example of pulling data from an external API
        inputs["extra_data"] = "This is extra data"
        return inputs
    @after_kickoff
    def log_results(self, output):
        # Example of logging results
        print(f"Results: {output}")
        return output
    @agent
    def retrieve_news(self) -> Agent:
        return Agent(
            config=self.agents_config["retrieve_news"],
            verbose=True,
        )
    @agent
    def website_scraper(self) -> Agent:
        return Agent(config=self.agents_config["website_scraper"], verbose=True)
    @agent
    def ai_news_writer(self) -> Agent:
        return Agent(config=self.agents_config["ai_news_writer"], verbose=True)
    @agent
    def file_writer(self) -> Agent:
        return Agent(config=self.agents_config["file_writer"], verbose=True)
    @task
    def retrieve_news_task(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_news_task"],
        )
    @task
    def website_scrape_task(self) -> Task:
        return Task(
            config=self.tasks_config["website_scrape_task"],
        )
    @task
    def ai_news_write_task(self) -> Task:
        return Task(
            config=self.tasks_config["ai_news_write_task"],
        )
    @task
    def file_write_task(self) -> Task:
        return Task(
            config=self.tasks_config["file_write_task"], output_file="news/{date}_report.md"
        )
    @crew
    def crew(self) -> Crew:
        """Creates the Agenticnewscurator crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )