class MockWebTask:
    def __init__(self, name, required_actions, hint_mode="explicit"):
        self.name = name
        self.required_actions = required_actions
        self.current_step = 0
        self.done = False
        self.hint_mode = hint_mode

    def reset(self):
        self.current_step = 0
        self.done = False
        return self._observation()

    def _observation(self):
        if self.current_step >= len(self.required_actions):
            next_action = "stop"
        else:
            next_action = self.required_actions[self.current_step]

        action_hints = {
            "search": "You are on the homepage. To begin, use search.",
            "click_product": "You are on the search results page. Open the product using click_product.",
            "add_to_cart": "You are on the product page. Add it to cart using add_to_cart.",
            "checkout": "You are on the cart page. Complete checkout using checkout.",
            "stop": "The task goal has been satisfied. Use stop."
        }

        if self.hint_mode == "explicit":
            state_hint = action_hints[next_action]

        elif self.hint_mode == "weak":
            page_names = {
                "search": "homepage",
                "click_product": "search results page",
                "add_to_cart": "product page",
                "checkout": "cart page",
                "stop": "goal completed page",
            }
            state_hint = f"You are on the {page_names[next_action]}."

        else:
            state_hint = (
                "You must infer the next action from the task progress and previous page state. "
                "No explicit next-action hint is provided."
            )

        return (
            f"Task: {self.name}\n"
            f"Current progress: step {self.current_step}/{len(self.required_actions)}\n"
            f"Page state: {state_hint}\n"
            f"Available actions: search, click_product, add_to_cart, checkout, go_back, stop"
        )

    def step(self, action):
        if self.done:
            return "Task already finished.", True, False

        expected = self.required_actions[self.current_step]

        if action == expected:
            self.current_step += 1
            if self.current_step == len(self.required_actions):
                self.done = True
                return "Success. Task completed.", True, True
            return self._observation(), False, True

        return (
            f"Error: expected action was {expected}, but agent did {action}.",
            False,
            False,
        )


def build_tasks(hint_mode="explicit"):
    base_actions = ["search", "click_product", "add_to_cart", "checkout"]

    tasks = []
    for horizon in [10, 20, 30, 40, 50]:
        actions = base_actions * ((horizon - 1) // len(base_actions) + 1)
        actions = actions[:horizon - 1] + ["stop"]

        tasks.append(
            MockWebTask(
                name=f"Long-horizon web task ({horizon} steps)",
                required_actions=actions,
                hint_mode=hint_mode,
            )
        )

    return tasks