from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class LLMAgent:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

        self.history = []
        self.valid_actions = [
            "click_product",
            "add_to_cart",
            "checkout",
            "search",
            "stop",
        ]

    def reset(self):
        self.history = []

    def next_action(self, observation):
        history_text = "\n".join(
            f"Step {i+1}: action={h['action']}"
            for i, h in enumerate(self.history[-5:])
        )
        
        prompt = f"""
You are controlling a web browser.

Your job is to choose the single best next action.

Previous actions:
{history_text if history_text else "None"}

Observation:
{observation}

The task follows this repeated workflow:
search -> click_product -> add_to_cart -> checkout

Important rules:
- If you are on the homepage and the task is not complete, choose search.
- If you are on the search results page, choose click_product.
- If you are on the product page, choose add_to_cart.
- If you are on the cart page, choose checkout.
- After checkout, the browser returns to the homepage and the workflow starts again.
- Only choose stop when the task goal is fully completed.
- Return exactly one valid action.
- Do not use markdown.
- Do not explain.

Valid actions:
search
click_product
add_to_cart
checkout
stop

Action:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # only decode newly generated tokens, not the full prompt
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print("RAW NEW OUTPUT:", repr(text))

        first_token = text.strip().split()[0] if text.strip() else ""

        if first_token in self.valid_actions:
            action = first_token
        else:
            action = "stop"

        print("PARSED ACTION:", action)

        self.history.append({"prompt": prompt, "action": action})
        return action