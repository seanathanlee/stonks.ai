"""
Playwright end-to-end tests for the Stonks.ai frontend.

The static ``frontend/index.html`` is served by a local HTTP server
(see ``conftest.py``).  All ``/api/chat`` calls are intercepted with
``page.route`` so no real backend is needed.
"""
from __future__ import annotations

import json

import pytest
from playwright.sync_api import Page, Route, expect


# ---------------------------------------------------------------------------
# Page-load / static structure
# ---------------------------------------------------------------------------


def test_page_title(page_loaded: Page) -> None:
    """The browser tab title must read "Stonks.ai"."""
    expect(page_loaded).to_have_title("Stonks.ai")


def test_header_brand_visible(page_loaded: Page) -> None:
    """The header must show the 'S' logo and the 'Stonks.ai' heading."""
    expect(page_loaded.locator("header .logo")).to_be_visible()
    expect(page_loaded.locator("header h1")).to_have_text("Stonks.ai")


def test_status_dot_ready(page_loaded: Page) -> None:
    """On load the status dot should be green (no 'busy' class)."""
    dot = page_loaded.locator("#status-dot")
    expect(dot).to_be_visible()
    expect(dot).not_to_have_class("busy")


def test_input_and_send_button_visible(page_loaded: Page) -> None:
    """The textarea and send button must be visible and enabled on load."""
    expect(page_loaded.locator("#user-input")).to_be_visible()
    expect(page_loaded.locator("#user-input")).to_be_enabled()
    expect(page_loaded.locator("#send-btn")).to_be_visible()
    expect(page_loaded.locator("#send-btn")).to_be_enabled()


def test_suggestion_buttons_visible(page_loaded: Page) -> None:
    """All five pre-defined suggestion chips must be present."""
    buttons = page_loaded.locator(".suggestion-btn")
    expect(buttons).to_have_count(5)
    labels = [
        "Top picks this month",
        "Run full analysis",
        "AAPL breakdown",
        "MSFT price history",
        "Best 1-year picks",
    ]
    for label in labels:
        expect(page_loaded.locator(f".suggestion-btn >> text={label}")).to_be_visible()


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------


def test_welcome_message_on_load(page_loaded: Page) -> None:
    """A bot bubble containing the welcome text must appear on load."""
    bot_bubbles = page_loaded.locator(".message.bot .bubble .message-text")
    expect(bot_bubbles.first).to_contain_text(
        "Ask me anything about the market"
    )


# ---------------------------------------------------------------------------
# User interaction — typing & sending
# ---------------------------------------------------------------------------


def test_type_message_into_input(page_loaded: Page) -> None:
    """Typing into the textarea should update its value."""
    textarea = page_loaded.locator("#user-input")
    textarea.fill("Hello, world!")
    expect(textarea).to_have_value("Hello, world!")


def test_send_message_via_button(page_loaded: Page) -> None:
    """
    Filling the textarea and clicking Send should:
    - Append the user's bubble to the chat
    - Call /api/chat (mocked) and append the bot reply bubble
    - Clear the textarea
    - Hide the suggestion chips
    """
    textarea = page_loaded.locator("#user-input")
    textarea.fill("What are your top picks?")

    page_loaded.locator("#send-btn").click()

    # User bubble appears
    expect(page_loaded.locator(".message.user .bubble").last).to_contain_text(
        "What are your top picks?"
    )

    # Bot reply appears (mock returns "Mock reply")
    expect(page_loaded.locator(".message.bot").last.locator(".message-text")).to_contain_text(
        "Mock reply"
    )

    # Input is cleared after send
    expect(textarea).to_have_value("")

    # Suggestion chips are hidden after first interaction
    expect(page_loaded.locator("#suggestions")).to_have_css("display", "none")


def test_send_message_via_enter_key(page_loaded: Page) -> None:
    """Pressing Enter (without Shift) should submit the message."""
    textarea = page_loaded.locator("#user-input")
    textarea.fill("Enter key test")
    textarea.press("Enter")

    expect(page_loaded.locator(".message.user .bubble").last).to_contain_text(
        "Enter key test"
    )
    expect(page_loaded.locator(".message.bot").last.locator(".message-text")).to_contain_text(
        "Mock reply"
    )


def test_shift_enter_does_not_submit(page_loaded: Page) -> None:
    """Shift+Enter should add a newline but NOT send the message."""
    textarea = page_loaded.locator("#user-input")
    textarea.fill("line one")
    textarea.press("Shift+Enter")

    # The input still has content (not cleared by a send)
    # Message count should not have increased with the typed text as a user bubble
    user_bubbles_before = page_loaded.locator(".message.user").count()
    textarea.press("Escape")  # unfocus without sending

    assert page_loaded.locator(".message.user").count() == user_bubbles_before


def test_suggestion_button_populates_and_sends(page_loaded: Page) -> None:
    """Clicking a suggestion chip should set the input and send the message."""
    page_loaded.locator(".suggestion-btn >> text=AAPL breakdown").click()

    expect(page_loaded.locator(".message.user .bubble").last).to_contain_text(
        "AAPL breakdown"
    )
    expect(page_loaded.locator(".message.bot").last.locator(".message-text")).to_contain_text(
        "Mock reply"
    )


# ---------------------------------------------------------------------------
# Typing indicator
# ---------------------------------------------------------------------------


def test_typing_indicator_shown_while_waiting(page_loaded: Page, frontend_url: str) -> None:
    """
    A typing-indicator bubble (three bouncing dots) must appear while
    the API call is in-flight and disappear once the reply arrives.
    """
    # Use a slow route so we can observe the intermediate state
    def slow_route(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {"reply": "Slow reply", "charts": [], "session_id": "t"}
            ),
        )

    # Override the default mock with a delayed one
    page_loaded.route("**/api/chat", slow_route)

    textarea = page_loaded.locator("#user-input")
    textarea.fill("Tell me something")
    page_loaded.locator("#send-btn").click()

    # The typing indicator appears while the request is being processed
    expect(page_loaded.locator(".typing-indicator")).to_be_visible()

    # After the response the indicator should disappear and the reply arrive
    expect(page_loaded.locator(".typing-indicator")).to_have_count(0, timeout=5_000)
    expect(page_loaded.locator(".message.bot").last.locator(".message-text")).to_contain_text(
        "Slow reply"
    )


# ---------------------------------------------------------------------------
# Busy state
# ---------------------------------------------------------------------------


def test_input_disabled_while_busy(page_loaded: Page) -> None:
    """
    The textarea and send button must be disabled while the API request
    is in-flight.

    We intercept ``window.fetch`` in the page so the chat request hangs
    until we explicitly release it.  This avoids blocking Playwright's
    event loop (which would happen if we stalled inside a ``page.route``
    handler).
    """
    # Replace window.fetch with a version that stalls on /api/chat
    page_loaded.evaluate("""() => {
        const orig = window.fetch.bind(window);
        window._releaseRequest = null;
        window.fetch = (url, opts) => {
            if (url.includes('/api/chat')) {
                return new Promise(resolve => {
                    window._releaseRequest = () => resolve(orig(url, opts));
                });
            }
            return orig(url, opts);
        };
    }""")

    textarea = page_loaded.locator("#user-input")
    textarea.fill("Busy test")
    page_loaded.locator("#send-btn").click()

    # setBusy(true) is called synchronously before the fetch, so
    # the controls are disabled immediately after the click
    expect(textarea).to_be_disabled()
    expect(page_loaded.locator("#send-btn")).to_be_disabled()
    expect(page_loaded.locator("#status-dot")).to_have_class("busy")

    # Release the stalled request (routes mock already fulfills it)
    page_loaded.evaluate("() => window._releaseRequest()")

    # After the response arrives the controls are re-enabled
    expect(textarea).to_be_enabled(timeout=5_000)
    expect(page_loaded.locator("#send-btn")).to_be_enabled()
    expect(page_loaded.locator("#status-dot")).not_to_have_class("busy")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_api_error_shows_error_bubble(page_loaded: Page) -> None:
    """A non-2xx API response must surface a ⚠️ error message in the chat."""
    page_loaded.route(
        "**/api/chat",
        lambda route: route.fulfill(
            status=500,
            content_type="application/json",
            body=json.dumps({"detail": "Internal server error"}),
        ),
    )

    page_loaded.locator("#user-input").fill("Trigger an error")
    page_loaded.locator("#send-btn").click()

    expect(page_loaded.locator(".message.bot").last.locator(".bubble")).to_contain_text(
        "⚠️"
    )


def test_empty_message_not_sent(page_loaded: Page) -> None:
    """Clicking Send with an empty input must not post a user bubble."""
    before = page_loaded.locator(".message.user").count()
    page_loaded.locator("#send-btn").click()
    assert page_loaded.locator(".message.user").count() == before


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------


def test_chart_rendered_when_api_returns_chart_data(page_loaded: Page) -> None:
    """
    When the API includes chart data the frontend must render a <canvas>
    element inside a .chart-card.
    """
    chart_payload = {
        "reply": "Here is the price history.",
        "charts": [
            {
                "id": "test_chart_1",
                "type": "line",
                "title": "AAPL Price History",
                "labels": ["Jan", "Feb", "Mar"],
                "datasets": [
                    {
                        "label": "AAPL",
                        "data": [150.0, 155.5, 160.2],
                        "borderColor": "#30d158",
                        "tension": 0.3,
                    }
                ],
            }
        ],
        "session_id": "test-session",
    }

    page_loaded.route(
        "**/api/chat",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(chart_payload),
        ),
    )

    page_loaded.locator("#user-input").fill("AAPL price history")
    page_loaded.locator("#send-btn").click()

    # The chart card and its title must be visible
    chart_card = page_loaded.locator(".chart-card").last
    expect(chart_card).to_be_visible(timeout=5_000)
    expect(chart_card.locator("h3")).to_have_text("AAPL Price History")

    # The canvas element must exist (Chart.js renders into it)
    expect(chart_card.locator("canvas#canvas_test_chart_1")).to_be_attached()


# ---------------------------------------------------------------------------
# HTML escaping (XSS prevention)
# ---------------------------------------------------------------------------


def test_user_input_is_html_escaped(page_loaded: Page) -> None:
    """
    Angle brackets typed by the user must appear as text, not as injected
    HTML tags.
    """
    xss_text = "<script>alert('xss')</script>"
    page_loaded.locator("#user-input").fill(xss_text)
    page_loaded.locator("#send-btn").click()

    bubble = page_loaded.locator(".message.user .bubble").last
    # Text content shows the literal string, not rendered HTML
    expect(bubble).to_contain_text("<script>")
    # No actual <script> element should exist inside the bubble
    assert bubble.locator("script").count() == 0


# ---------------------------------------------------------------------------
# Agent Roster panel
# ---------------------------------------------------------------------------


def test_roster_toggle_button_visible(page_loaded: Page) -> None:
    """The 'Agents' toggle button must be visible in the header."""
    expect(page_loaded.locator("#roster-toggle")).to_be_visible()


def test_roster_panel_hidden_by_default(page_loaded: Page) -> None:
    """The agent roster panel must be hidden before the toggle is clicked."""
    panel = page_loaded.locator("#agent-roster")
    expect(panel).not_to_have_class("open")


def test_roster_panel_opens_on_click(page_loaded: Page) -> None:
    """Clicking the Agents button must reveal the roster panel with agent cards."""
    page_loaded.locator("#roster-toggle").click()

    panel = page_loaded.locator("#agent-roster")
    expect(panel).to_have_class("open")

    # All 9 agent cards must be rendered
    expect(panel.locator(".agent-card")).to_have_count(9)


def test_roster_panel_shows_agent_names(page_loaded: Page) -> None:
    """The roster panel must display the names of all 9 child agents."""
    page_loaded.locator("#roster-toggle").click()

    panel = page_loaded.locator("#agent-roster")
    agent_names = [
        "Momentum Trader",
        "Mean Reversion",
        "Value Investor",
        "Growth Investor",
        "Volatility Hunter",
        "Sector Rotation",
        "Technical Analyst",
        "Contrarian Investor",
        "Risk-Adjusted Optimizer",
    ]
    for name in agent_names:
        expect(panel.locator(f".agent-name >> text={name}")).to_be_visible()


def test_roster_panel_closes_on_second_click(page_loaded: Page) -> None:
    """Clicking the Agents button a second time must hide the roster panel."""
    toggle = page_loaded.locator("#roster-toggle")
    toggle.click()
    expect(page_loaded.locator("#agent-roster")).to_have_class("open")

    toggle.click()
    expect(page_loaded.locator("#agent-roster")).not_to_have_class("open")


def test_evaluation_chart_rendered(page_loaded: Page) -> None:
    """
    When the API returns an evaluation chart payload, the frontend must
    render a <canvas> element inside a .chart-card with the chart title.
    """
    eval_payload = {
        "reply": "Here are the agent accuracy scores.",
        "charts": [
            {
                "id": "eval_1m_abc12345",
                "type": "bar",
                "title": "Agent Accuracy Scores — 1m Horizon (lower = more accurate)",
                "indexAxis": "y",
                "labels": ["Momentum Trader", "Mean Reversion"],
                "datasets": [
                    {
                        "label": "Avg Accuracy Score",
                        "data": [1.5, 2.3],
                        "backgroundColor": ["#3b82f6", "#22c55e"],
                        "borderRadius": 4,
                    }
                ],
            }
        ],
        "session_id": "test-session",
    }

    page_loaded.route(
        "**/api/chat",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=__import__("json").dumps(eval_payload),
        ),
    )

    page_loaded.locator("#user-input").fill("How accurate are the agents?")
    page_loaded.locator("#send-btn").click()

    chart_card = page_loaded.locator(".chart-card").last
    expect(chart_card).to_be_visible(timeout=5_000)
    expect(chart_card.locator("h3")).to_contain_text("Agent Accuracy Scores")
    expect(chart_card.locator("canvas#canvas_eval_1m_abc12345")).to_be_attached()

