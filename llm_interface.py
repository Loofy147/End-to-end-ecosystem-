import os
import openai
from anthropic import Anthropic
from config import Config
import logging

logging.basicConfig(level=Config.LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")

class LLMInterface:
    def __init__(self):
        self.provider = Config.LLM_PROVIDER
        self.openai_client = None
        self.anthropic_client = None
        self.call_count = 0

        if self.provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in config.")
            self.openai_client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
        elif self.provider == "anthropic":
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY is not set in config.")
            self.anthropic_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _get_discount(self) -> float:
        if self.call_count >= Config.DISCOUNT_14_PERCENT_THRESHOLD:
            return 0.14
        elif self.call_count >= Config.DISCOUNT_7_PERCENT_THRESHOLD:
            return 0.07
        elif self.call_count >= Config.DISCOUNT_3_PERCENT_THRESHOLD:
            return 0.03
        return 0.0

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        # This is a simplified cost model. Real costs vary by model and provider.
        # Example: OpenAI GPT-4o pricing (as of early 2024, check current pricing)
        # Input: $5.00 / 1M tokens, Output: $15.00 / 1M tokens
        # Anthropic Claude 3 Opus: Input: $15.00 / 1M tokens, Output: $75.00 / 1M tokens

        cost_per_million_input = 0.0
        cost_per_million_output = 0.0

        if self.provider == "openai":
            cost_per_million_input = 5.00
            cost_per_million_output = 15.00
        elif self.provider == "anthropic":
            cost_per_million_input = 15.00
            cost_per_million_output = 75.00

        input_cost = (prompt_tokens / 1_000_000) * cost_per_million_input
        output_cost = (completion_tokens / 1_000_000) * cost_per_million_output
        total_cost = input_cost + output_cost

        discount = self._get_discount()
        discounted_cost = total_cost * (1 - discount)
        return discounted_cost

    def generate_text(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Tuple[str, float]:
        self.call_count += 1
        model_name = ""
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        cost = 0.0

        try:
            if self.provider == "openai":
                model_name = Config.OPENAI_MODEL
                response = self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS,
                    temperature=temperature if temperature is not None else Config.LLM_TEMPERATURE,
                )
                response_text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens

            elif self.provider == "anthropic":
                model_name = Config.ANTHROPIC_MODEL
                response = self.anthropic_client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS,
                    temperature=temperature if temperature is not None else Config.LLM_TEMPERATURE,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                )
                response_text = response.content[0].text
                # Anthropic API doesn't directly provide token counts in the same way as OpenAI for messages.create
                # This is a placeholder; you might need to estimate or use a different API for precise counts.
                prompt_tokens = len(prompt.split()) # Rough estimate
                completion_tokens = len(response_text.split()) # Rough estimate

            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            logging.info(f"LLM Call ({self.provider}/{model_name}) - Tokens: P:{prompt_tokens}, C:{completion_tokens} - Cost: ${cost:.4f}")
            return response_text, cost

        except Exception as e:
            logging.error(f"Error during LLM call to {self.provider}: {e}")
            return f"Error: {e}", 0.0

# Example usage (for testing):
if __name__ == "__main__":
    # Ensure .env is set up with API keys
    # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    # os.environ["LLM_PROVIDER"] = "openai"

    llm_interface = LLMInterface()
    
    print("\n--- Testing basic generation ---")
    response, cost = llm_interface.generate_text("What is the capital of France?")
    print(f"Response: {response}\nCost: ${cost:.4f}")

    print("\n--- Testing with specific parameters ---")
    response, cost = llm_interface.generate_text("Write a short poem about AI.", max_tokens=50, temperature=0.9)
    print(f"Response: {response}\nCost: ${cost:.4f}")

    print("\n--- Testing discount tiers (simulated) ---")
    # Simulate many calls to hit discount tiers
    llm_interface.call_count = Config.DISCOUNT_3_PERCENT_THRESHOLD - 5
    response, cost = llm_interface.generate_text("Test discount 3%.")
    print(f"Call count: {llm_interface.call_count}, Cost: ${cost:.4f}")

    llm_interface.call_count = Config.DISCOUNT_7_PERCENT_THRESHOLD - 5
    response, cost = llm_interface.generate_text("Test discount 7%.")
    print(f"Call count: {llm_interface.call_count}, Cost: ${cost:.4f}")

    llm_interface.call_count = Config.DISCOUNT_14_PERCENT_THRESHOLD - 5
    response, cost = llm_interface.generate_text("Test discount 14%.")
    print(f"Call count: {llm_interface.call_count}, Cost: ${cost:.4f}")




