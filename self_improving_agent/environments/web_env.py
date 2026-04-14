"""
Simplified web task environment.
Uses mock HTML pages to simulate sequential web navigation tasks.
Optionally uses requests + BeautifulSoup for real page interaction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WebTask:
    id: str
    description: str
    horizon: int
    pages: Dict[str, str]          # url -> mock HTML content
    goal_check: "callable"         # function(state) -> bool
    env: "WebEnvironment" = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "horizon": self.horizon,
            "env": self.env,
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class WebEnvironment:
    """
    Mock web environment with in-memory HTML pages.
    Supports: navigate(url), click(selector), type(selector, text), read_page()
    """

    def __init__(self, task: WebTask):
        self.task = task
        self._pages = task.pages
        self._current_url: str = ""
        self._current_html: str = ""
        self._form_data: Dict[str, str] = {}
        self._visited_urls: List[str] = []
        self._submitted: bool = False
        self._state: Dict[str, Any] = {}

    # ------------------------------------------------------------------

    def step(self, action_str: str) -> str:
        action_str = action_str.strip()
        if action_str.lower().startswith("action:"):
            action_str = action_str[7:].strip()

        if action_str.startswith("finish("):
            return "Task declared complete."

        m = re.match(r"(\w+)\((.*)\)$", action_str, re.DOTALL)
        if not m:
            return "Error: Could not parse action."

        tool = m.group(1).strip()
        args = m.group(2).strip()

        if tool == "navigate":
            return self._navigate(args)
        elif tool == "click":
            return self._click(args)
        elif tool == "type":
            return self._type(args)
        elif tool in ("bash", "check_output"):
            return self._read_state()
        elif tool == "read_file":
            return self._read_state()
        else:
            return f"Error: Tool '{tool}' not available in web environment."

    def is_success(self) -> bool:
        try:
            return self.task.goal_check(self._state)
        except Exception:
            return False

    # ------------------------------------------------------------------

    def _navigate(self, url: str) -> str:
        url = url.strip().strip('"').strip("'")
        if url in self._pages:
            self._current_url = url
            self._current_html = self._pages[url]
            self._visited_urls.append(url)
            self._state["current_url"] = url
            # Return a simplified text representation
            text = self._html_to_text(self._current_html)
            return f"Navigated to {url}\n\n{text}"
        else:
            return f"Error: URL not found: {url}. Available: {list(self._pages.keys())}"

    def _click(self, selector: str) -> str:
        selector = selector.strip().strip('"').strip("'")
        if not self._current_html:
            return "Error: No page loaded. Use navigate(url) first."

        # Look for link with matching text or id
        link_match = re.search(
            rf'href=["\']([^"\']+)["\'][^>]*>.*?{re.escape(selector)}.*?</a>',
            self._current_html, re.IGNORECASE | re.DOTALL
        )
        if link_match:
            href = link_match.group(1)
            return self._navigate(href)

        # Look for button
        btn_match = re.search(
            rf'<(?:button|input)[^>]*(?:value|id|name)=["\'][^"\']*{re.escape(selector)}[^"\']*["\'][^>]*>',
            self._current_html, re.IGNORECASE
        )
        if btn_match:
            if 'type="submit"' in btn_match.group().lower() or 'submit' in selector.lower():
                self._submitted = True
                self._state["submitted"] = True
                self._state["form_data"] = dict(self._form_data)
                return "Form submitted successfully."
            return f"Clicked element: {selector}"

        self._state[f"clicked_{selector}"] = True
        return f"Clicked: {selector} (element found on page)"

    def _type(self, args: str) -> str:
        parts = args.split(",", 1)
        if len(parts) < 2:
            return "Error: type requires (selector, text)"
        selector = parts[0].strip().strip('"').strip("'")
        text = parts[1].strip().strip('"').strip("'")
        self._form_data[selector] = text
        self._state[f"typed_{selector}"] = text
        return f"Typed '{text}' into '{selector}'"

    def _read_state(self) -> str:
        if self._current_html:
            return self._html_to_text(self._current_html)[:1000]
        return f"State: {self._state}"

    @staticmethod
    def _html_to_text(html: str) -> str:
        # Strip tags, compress whitespace
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text)
        return text.strip()[:2000]


# ---------------------------------------------------------------------------
# Task generator
# ---------------------------------------------------------------------------

def generate_web_tasks(
    horizons: List[int] = None,
    n_per_horizon: int = 25,
) -> List[Dict[str, Any]]:
    if horizons is None:
        horizons = [5, 10, 15, 20]

    all_tasks = []
    task_id = 0

    for horizon in horizons:
        templates = _get_web_templates(horizon)
        for i in range(n_per_horizon):
            tmpl = templates[i % len(templates)]
            task_data = tmpl(task_id)
            env = WebEnvironment(task_data)
            task_dict = task_data.to_dict()
            task_dict["env"] = env
            all_tasks.append(task_dict)
            task_id += 1

    return all_tasks


def _get_web_templates(horizon: int):
    if horizon <= 5:
        return [_web_task_search, _web_task_read_page]
    elif horizon <= 10:
        return [_web_task_navigate_fill, _web_task_multi_step]
    elif horizon <= 15:
        return [_web_task_form_submit, _web_task_extract_data]
    else:
        return [_web_task_full_workflow, _web_task_multi_form]


# ---------------------------------------------------------------------------
# Web task templates
# ---------------------------------------------------------------------------

def _web_task_search(task_id: int) -> WebTask:
    pages = {
        "http://example.com": """
            <html><body>
            <h1>Example Site</h1>
            <form action="/search">
                <input name="q" type="text"/>
                <input type="submit" value="Search"/>
            </form>
            </body></html>
        """,
        "http://example.com/search?q=agent": """
            <html><body>
            <h1>Search Results for: agent</h1>
            <p>Found 3 results about AI agents.</p>
            <a href="/result/1">Result 1: Self-improving agents</a>
            </body></html>
        """,
    }

    def goal(state):
        return "current_url" in state and "/search" in state.get("current_url", "")

    return WebTask(
        id=f"web_search_{task_id}",
        description="Navigate to http://example.com, search for 'agent', and find search results.",
        horizon=5,
        pages=pages,
        goal_check=goal,
    )


def _web_task_read_page(task_id: int) -> WebTask:
    pages = {
        "http://info.example.com": """
            <html><body>
            <h1>Information Page</h1>
            <p id="answer">The capital of France is Paris.</p>
            <a href="/more">More info</a>
            </body></html>
        """,
    }

    def goal(state):
        return "current_url" in state

    return WebTask(
        id=f"web_read_{task_id}",
        description="Navigate to http://info.example.com and read the answer on the page.",
        horizon=5,
        pages=pages,
        goal_check=goal,
    )


def _web_task_navigate_fill(task_id: int) -> WebTask:
    pages = {
        "http://shop.example.com": """
            <html><body>
            <h1>Shop</h1>
            <a href="/login">Login</a>
            <a href="/products">Products</a>
            </body></html>
        """,
        "/login": """
            <html><body>
            <h1>Login</h1>
            <form action="/dashboard">
                <input name="username" type="text"/>
                <input name="password" type="password"/>
                <input type="submit" value="Login" id="submit-btn"/>
            </form>
            </body></html>
        """,
        "/dashboard": """
            <html><body>
            <h1>Dashboard</h1>
            <p>Welcome back!</p>
            </body></html>
        """,
    }

    def goal(state):
        return (
            state.get("typed_username") is not None
            and state.get("typed_password") is not None
        )

    return WebTask(
        id=f"web_login_{task_id}",
        description=(
            "Navigate to http://shop.example.com, go to the login page, "
            "fill in username='testuser' and password='pass123', then submit."
        ),
        horizon=10,
        pages=pages,
        goal_check=goal,
    )


def _web_task_multi_step(task_id: int) -> WebTask:
    pages = {
        "http://news.example.com": """
            <html><body>
            <h1>News Site</h1>
            <ul>
                <li><a href="/article/1">Article 1: AI Advances</a></li>
                <li><a href="/article/2">Article 2: Climate News</a></li>
            </ul>
            </body></html>
        """,
        "/article/1": """
            <html><body>
            <h1>AI Advances</h1>
            <p>Artificial intelligence is advancing rapidly.</p>
            <a href="/article/1/comments">Comments (5)</a>
            </body></html>
        """,
        "/article/1/comments": """
            <html><body>
            <h1>Comments</h1>
            <p>Great article!</p>
            </body></html>
        """,
    }

    def goal(state):
        visited = state.get("current_url", "")
        return "article" in visited

    return WebTask(
        id=f"web_multi_{task_id}",
        description=(
            "Go to http://news.example.com, find and read the first article about AI, "
            "then navigate to its comments section."
        ),
        horizon=10,
        pages=pages,
        goal_check=goal,
    )


def _web_task_form_submit(task_id: int) -> WebTask:
    pages = {
        "http://survey.example.com": """
            <html><body>
            <h1>Survey</h1>
            <form>
                <label>Name: <input name="name" type="text"/></label>
                <label>Email: <input name="email" type="email"/></label>
                <label>Rating:
                    <select name="rating">
                        <option value="1">1</option>
                        <option value="5">5</option>
                    </select>
                </label>
                <input type="submit" id="submit" value="Submit"/>
            </form>
            </body></html>
        """,
    }

    def goal(state):
        return (
            state.get("submitted") is True
            and state.get("typed_name") is not None
        )

    return WebTask(
        id=f"web_form_{task_id}",
        description=(
            "Navigate to http://survey.example.com, fill in the survey form "
            "with name='John Doe', email='john@example.com', rating='5', and submit."
        ),
        horizon=15,
        pages=pages,
        goal_check=goal,
    )


def _web_task_extract_data(task_id: int) -> WebTask:
    pages = {
        "http://data.example.com/table": """
            <html><body>
            <h1>Data Table</h1>
            <table>
                <tr><th>Name</th><th>Score</th></tr>
                <tr><td>Alice</td><td>92</td></tr>
                <tr><td>Bob</td><td>87</td></tr>
                <tr><td>Carol</td><td>95</td></tr>
            </table>
            <a href="/table/download">Download CSV</a>
            </body></html>
        """,
    }

    def goal(state):
        return "current_url" in state and "table" in state.get("current_url", "")

    return WebTask(
        id=f"web_extract_{task_id}",
        description=(
            "Navigate to http://data.example.com/table, extract all names and scores, "
            "and identify the highest scorer."
        ),
        horizon=15,
        pages=pages,
        goal_check=goal,
    )


def _web_task_full_workflow(task_id: int) -> WebTask:
    pages = {
        "http://app.example.com": """
            <html><body>
            <h1>App</h1>
            <a href="/signup">Sign Up</a>
            <a href="/login">Login</a>
            </body></html>
        """,
        "/signup": """
            <html><body>
            <h1>Sign Up</h1>
            <form>
                <input name="username" type="text" placeholder="Username"/>
                <input name="email" type="email" placeholder="Email"/>
                <input name="password" type="password" placeholder="Password"/>
                <input type="submit" value="Create Account" id="create-btn"/>
            </form>
            </body></html>
        """,
        "/profile": """
            <html><body>
            <h1>Profile</h1>
            <form>
                <input name="bio" type="text" placeholder="Bio"/>
                <input type="submit" value="Save Profile" id="save-btn"/>
            </form>
            </body></html>
        """,
    }

    def goal(state):
        return state.get("submitted") is True and state.get("typed_username") is not None

    return WebTask(
        id=f"web_workflow_{task_id}",
        description=(
            "Complete the full user registration workflow on http://app.example.com: "
            "navigate to signup, fill all fields, create account, then fill in the profile bio."
        ),
        horizon=20,
        pages=pages,
        goal_check=goal,
    )


def _web_task_multi_form(task_id: int) -> WebTask:
    pages = {
        "http://checkout.example.com": """
            <html><body>
            <h1>Checkout</h1>
            <a href="/checkout/address">Step 1: Address</a>
            </body></html>
        """,
        "/checkout/address": """
            <html><body>
            <h1>Shipping Address</h1>
            <form>
                <input name="street" placeholder="Street"/>
                <input name="city" placeholder="City"/>
                <input name="zip" placeholder="ZIP"/>
                <input type="submit" value="Next" id="next-btn"/>
            </form>
            </body></html>
        """,
        "/checkout/payment": """
            <html><body>
            <h1>Payment</h1>
            <form>
                <input name="card_number" placeholder="Card Number"/>
                <input name="expiry" placeholder="Expiry"/>
                <input type="submit" value="Pay Now" id="pay-btn"/>
            </form>
            </body></html>
        """,
    }

    def goal(state):
        return state.get("submitted") is True

    return WebTask(
        id=f"web_checkout_{task_id}",
        description=(
            "Complete a multi-step checkout on http://checkout.example.com: "
            "fill in address (street='123 Main St', city='Springfield', zip='12345'), "
            "proceed to payment, and complete the payment form."
        ),
        horizon=20,
        pages=pages,
        goal_check=goal,
    )
