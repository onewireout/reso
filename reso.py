#!/usr/bin/env python3
from collections import deque
import glob
import json
import os
import threading
import time
import requests
from rich import box
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, OptionList, RichLog, TextArea
from textual.widgets.option_list import Option

import tools
from tools import LLAMA_URL, MODEL, TOOL_SCHEMAS, call_tool
from render import (
    COLOR_CODE,
    COLOR_LINK,
    COLOR_MODEL,
    COLOR_REASONING,
    COLOR_REASONING_TEXT,
    COLOR_USER,
    MD_TABLE_SEP,
    render_block_line,
    render_banner,
    render_markdown_line,
    render_pencil_line,
    split_table_row,
)

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sessions")


def preview_value(v, limit: int = 60) -> str:
    text = repr(v)
    if len(text) > limit:
        return text[:limit] + "...'"
    return text


def shorten(text: str, limit: int = 100) -> str:
    """Squash to one line for the approval prompt."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_tool_target(name: str, arguments: dict) -> str:
    if name == "read_file":
        path = arguments.get("path", "?")
        start, end = arguments.get("start_line"), arguments.get("end_line")
        if start or end:
            return f"{path} (lines {start or 1}-{end or 'end'})"
        return path
    if name == "search_files":
        pattern = arguments.get("pattern", "?")
        path = arguments.get("path", ".")
        return f'"{pattern}" in {path}'
    if name == "glob_files":
        pattern = arguments.get("pattern", "?")
        path = arguments.get("path", ".")
        return f'"{pattern}" under {path}'
    if name == "list_directory":
        return arguments.get("path", ".")
    if name == "get_file_info":
        return arguments.get("path", "?")
    if name == "write_file":
        return arguments.get("path", "?")
    if name == "edit_file":
        return arguments.get("path", "?")
    if name == "replace_lines":
        path = arguments.get("path", "?")
        start, end = arguments.get("start_line", "?"), arguments.get("end_line", "?")
        return f"{path} (lines {start}-{end})"
    if name == "move_file":
        return f"{arguments.get('src', '?')} -> {arguments.get('dst', '?')}"
    if name == "make_directory":
        return arguments.get("path", "?")
    if name == "outline_file":
        return arguments.get("path", "?")
    if name == "delete_file":
        return arguments.get("path", "?")
    if name == "multi_edit":
        edits = arguments.get("edits") or []
        return f"{arguments.get('path', '?')} ({len(edits)} edits)"
    if name == "remember":
        return shorten(arguments.get("note", "?"))
    if name == "web_search":
        return f'"{arguments.get("query", "?")}"'
    if name == "fetch_url":
        return arguments.get("url", "?")
    if name == "run_command":
        return shorten(arguments.get("command", "?"), 200)
    return shorten(", ".join(f"{k}={v!r}" for k, v in arguments.items()) or "?")


class MessageInput(TextArea):
    class Submitted(Message):
        pass

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted())
        elif event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")
        else:
            await super()._on_key(event)


MAX_CONTEXT = tools.MODEL_CONTEXTS[tools.MODEL]
COMPACT_AT = MAX_CONTEXT // 2
MAX_TOOL_OUTPUT = 16000
MAX_SESSIONS = 10
VALID_EFFORTS = ("low", "medium", "xhigh")


def format_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def estimate_tokens(messages: list) -> int:
    """Rough token count, ~4 chars per token."""
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        for call in m.get("tool_calls") or []:
            chars += len(json.dumps(call.get("function", {}).get("arguments", {})))
    return chars // 4


def cap_tool_output(text: str) -> str:
    """Cap tool output so one big result can't flood the context."""
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    note = (
        f"[truncated: showing first {MAX_TOOL_OUTPUT} of {len(text)} chars; "
        "the rest can be re-read or re-run — only if you need it]"
    )
    return f"{note}\n{text[:MAX_TOOL_OUTPUT]}\n{note}"

class Reso(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #display {
        height: 1fr;
        scrollbar-size: 0 0;
    }
    #log {
        height: auto;
        background: transparent;
        overflow-y: hidden;
        scrollbar-size: 0 0;
    }
    #live {
        height: auto;
    }
    #footer {
        dock: bottom;
        height: auto;
    }
    #input {
        border: none;
        border-top: solid #D4B9A8;
        border-bottom: solid #D4B9A8;
        background: transparent;
        padding: 0 1;
        height: auto;
        max-height: 10;
    }
    #input.-command {
        color: $text-muted;
    }
    #input .text-area--cursor {
        background: #BDB8B0;
    }
    #statusbar {
        height: 1;
        layout: horizontal;
    }
    #status {
        color: #121218;
        background: #8C9B91;
        height: 1;
        width: auto;
    }
    #tokens {
        color: #121218;
        background: #D4AF37;
        height: 1;
        width: auto;
        margin-left: 1;
    }
    #speed {
        color: #121218;
        background: #6FAFB2;
        height: 1;
        width: auto;
        margin-left: 1;
    }
    #speed.idle {
        background: #3D5A5E;
    }
    #compact-bar {
        height: 1;
        width: 100%;
        padding: 0 1;
        display: none;
    }
    #effort-list {
        height: auto;
        max-height: 5;
        border: solid $accent;
    }
    #resume-list {
        height: auto;
        max-height: 5;
        border: solid $accent;
    }
    #approval-list {
        height: auto;
        max-height: 5;
        border: solid $warning;
    }
    #mtp-list {
        height: auto;
        max-height: 5;
        border: solid $accent;
    }
    """

    BINDINGS = [("escape", "interrupt", "Stop")]

    def __init__(self):
        super().__init__()
        self.messages = [self.build_system_message()]
        self.reasoning_effort = "medium"
        self._pending_approval = None
        self._live_text = ""
        self._live_style = None
        self._live_markdown = False
        self._in_fence = False
        self._fence_start = None
        self._table_start = None
        self._history: list = []
        self._token_baseline = 0
        self._chars_since_sync = 0
        self._interrupt = threading.Event()
        self._generating = False
        self._compact_notice = False
        self._compact_notice_pending = None
        self._compacting = False
        self._compact_chars = 0
        self._speed_samples: deque = deque()
        self._speed_chars = 0
        self._speed_last_rate = 0.0
        self._pencil_active = False
        self._pencil_pos = 0.0
        self._pencil_timer = None

        # one session file per run, named by start time
        self.session_started = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.session_path = os.path.join(SESSIONS_DIR, f"{self.session_started}.json")

        tools.PROGRESS = self.report_tool_progress
        tools.STREAM = self.report_tool_stream
        tools.SHOULD_STOP = self._interrupt.is_set
        tools.EFFORT = self.reasoning_effort
        tools.COMPACT = self.compact_context


    def build_system_message(self) -> dict:
        """System message with the working dir, date and current notes."""
        notes = tools.read_notes()
        return {"role": "system", "content": (
            "System prompt goes here.\n\n"
            f"Working directory: {os.getcwd()}\n"
            f"Today's date: {time.strftime('%Y-%m-%d')}\n\n"
            "Notes saved with the remember tool persist between sessions "
            "and appear here at the start of each one."
            + (
                f"\n\n{notes}\n\n"
                "These are for your own reference; no need to bring them up "
                "unless they're relevant."
                if notes else ""
            )
        )}

    def action_interrupt(self) -> None:
        if not self._generating:
            return
        self._interrupt.set()
        if self._pending_approval is not None:
            self._pending_approval["approved"] = False
            self._pending_approval["event"].set()

    def prune_sessions(self) -> None:
        """Delete the oldest sessions past MAX_SESSIONS. Never the current one."""
        try:
            paths = sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.json")))
            excess = len(paths) - MAX_SESSIONS + 1
            for path in paths[: max(0, excess)]:
                if path != self.session_path:
                    os.remove(path)
        except OSError:
            pass

    def save_session(self) -> None:
        """Save the conversation to the session file."""
        user_msgs = [
            m for m in self.messages
            if m["role"] == "user" and isinstance(m.get("content"), str)
        ]
        title = " ".join(user_msgs[0]["content"].split())[:80] if user_msgs else ""
        payload = {
            "started": self.session_started,
            "updated": time.strftime("%Y-%m-%d_%H-%M-%S"),
            "effort": self.reasoning_effort,
            "title": title,
            "tokens": estimate_tokens(self.messages),
            "messages": self.messages,
        }
        try:
            os.makedirs(SESSIONS_DIR, exist_ok=True)
            tmp_path = self.session_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.session_path)
            self.prune_sessions()
        except OSError:
            pass

    def on_unmount(self) -> None:
        # Textual has no on_exit, this is the one that fires
        self.save_session()

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="display"):
            yield RichLog(id="log", markup=True, wrap=True, auto_scroll=False, highlight=False, min_width=1)
            yield Static("", id="live")
        with Vertical(id="footer"):
            yield Static("", id="compact-bar")
            yield MessageInput(id="input", show_line_numbers=False, soft_wrap=True)
            with Horizontal(id="statusbar"):
                yield Static(f"Reasoning: {self.reasoning_effort}", id="status")
                yield Static(f"{format_tokens(0)}/{format_tokens(MAX_CONTEXT)}", id="tokens")
                yield Static("", id="speed")

    def on_mount(self) -> None:
        # banner goes in the log so it scrolls away with everything else
        banner = render_banner()
        self._history.append(banner)
        self.query_one("#log", RichLog).write(banner, expand=True)
        self.query_one("#input", MessageInput).focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Dim the input while it's a /command."""
        input_field = self.query_one("#input", MessageInput)
        if event.text_area is not input_field:
            return
        input_field.set_class(bool(event.text_area.text.startswith("/")), "-command")

    def on_message_input_submitted(self, event: MessageInput.Submitted) -> None:
        self.submit_message()

    def submit_message(self) -> None:
        text_area = self.query_one("#input", MessageInput)
        message = text_area.text.strip()
        if not message:
            return
        if message == "/effort":
            text_area.text = ""
            self.show_effort_picker()
            return

        if message == "/mtp":
            text_area.text = ""
            self.show_mtp_picker()
            return

        if message == "/resume":
            # a running worker would keep appending to the old message list
            if self._generating:
                self.write_line("Still generating — press escape first.", "dim")
                return
            text_area.text = ""
            self.show_resume_picker()
            return

        # one turn at a time, two workers would mix up the message list
        if self._generating:
            self.write_line("Still generating — press escape to stop.", "dim")
            return

        text_area.text = ""

        line = Text("You: ", style=f"bold {COLOR_USER}")
        line.append(message, style=COLOR_USER)
        line.append("\n")
        self.flush_live()
        self._commit_to_log(line)
        self.messages.append({"role": "user", "content": message})
        # set early so a fast double-enter can't start two turns
        self._generating = True
        self.run_worker(self.stream_response, thread=True)

    def update_token_count(self, total: int, estimate: bool = False) -> None:
        self._token_baseline = total
        self._chars_since_sync = 0
        prefix = "~" if estimate else ""
        self.query_one("#tokens", Static).update(f"{prefix}{format_tokens(total)}/{format_tokens(MAX_CONTEXT)}")
        self._maybe_notify_compaction()

    # ~1.5k token summary, ~4 chars per token
    COMPACT_TARGET_CHARS = 6000

    def start_compact_bar(self) -> None:
        """Show the compact bar at 0%."""
        self._compacting = True
        self._compact_chars = 0
        self._draw_compact_bar(0.0)

    def advance_compact_bar(self, chars: int) -> None:
        """Update the compact bar as the summary streams. Stops short of 100%."""
        if not self._compacting:
            return
        self._compact_chars += chars
        progress = min(0.99, self._compact_chars / self.COMPACT_TARGET_CHARS)
        self._draw_compact_bar(progress)

    def finish_compact_bar(self) -> None:
        """Fill the bar, then hide it a moment later."""
        self._compacting = False
        self._draw_compact_bar(1.0)
        self.set_timer(1.2, self.hide_compact_bar)

    def hide_compact_bar(self) -> None:
        self.query_one("#compact-bar", Static).display = False

    def abort_compact_bar(self) -> None:
        """Compact never finished. Drop the bar and say so."""
        if not self._compacting:
            return
        self._compacting = False
        self.hide_compact_bar()
        self.write_line("Compaction failed, context unchanged.", "dim")

    def _draw_compact_bar(self, progress: float) -> None:
        bar = self.query_one("#compact-bar", Static)
        bar.display = True
        bar.update(self._render_bar(progress))

    def _render_bar(self, progress: float) -> Text:
        width = 30
        filled = round(progress * width)
        text = Text()
        text.append("compacting: ", style=f"bold {COLOR_MODEL}")
        text.append("█" * filled, style=COLOR_MODEL)
        text.append("░" * (width - filled), style=f"dim {COLOR_MODEL}")
        text.append(f" {int(progress * 100)}%", style=f"dim {COLOR_MODEL}")
        return text

    def note_chars(self, count: int) -> None:
        self._chars_since_sync += count
        estimate = self._token_baseline + self._chars_since_sync // 4
        self.update_token_count(estimate, estimate=True)

    def _reset_speed(self) -> None:
        # keep the last number up while the new stream comes in
        self.query_one("#speed", Static).remove_class("idle")
        self._speed_samples.clear()
        self._speed_chars = 0
        self._speed_last_rate = 0.0

    def _dim_speed(self) -> None:
        self.query_one("#speed", Static).add_class("idle")

    def note_stream_chars(self, count: int) -> None:
        """Update the tok/s readout, smoothed over ~3 seconds."""
        self._speed_chars += count
        now = time.monotonic()
        self._speed_samples.append((now, self._speed_chars))
        while len(self._speed_samples) > 2 and now - self._speed_samples[0][0] >= 3:
            self._speed_samples.popleft()
        if len(self._speed_samples) < 2:
            return
        t0, c0 = self._speed_samples[0]
        dt = now - t0
        # skip short windows, early rates are noisy
        if dt < 1.5:
            return
        rate = (self._speed_chars - c0) / dt / 4  # ~4 chars per token
        # only repaint on real changes so the label doesn't flicker
        if self._speed_last_rate and abs(rate - self._speed_last_rate) < self._speed_last_rate * 0.1:
            return
        self._speed_last_rate = rate
        self.query_one("#speed", Static).update(f"{int(rate)} tok/s")

    def _maybe_notify_compaction(self) -> None:
        """Flag a compact notice. The worker adds it between rounds so it can't land mid tool call."""
        if self._compact_notice or self._token_baseline < COMPACT_AT:
            return
        self._compact_notice = True
        self._compact_notice_pending = ({
            "role": "user",
            "content": (
                f"[Context at {format_tokens(self._token_baseline)}/"
                f"{format_tokens(MAX_CONTEXT)}. When you reach a natural break "
                "in the current task: write a terse work-note (terse checklist "
                "of completed steps) to notes/ with a pointer in NOTES.md, "
                "then call compact() with a summary of ~1.5k tokens covering "
                "the goal, what has been done, and what is next.]"
            ),
        })

    def compact_context(self, summary: str) -> str:
        """Called by the compact tool. Swaps the conversation for the summary."""
        if not summary or not summary.strip():
            return "Error: empty summary, nothing compacted."
        summary = " ".join(summary.split())
        done = threading.Event()

        def apply():
            # the worker is waiting on done.wait(), so always signal, even on errors
            try:
                self.finish_compact_bar()
                self.messages = [
                    self.build_system_message(),
                    {"role": "user", "content": summary},
                ]
                self._compact_notice = False
                self._compact_notice_pending = None
                self._chars_since_sync = 0
                self.update_token_count(estimate_tokens(self.messages), estimate=True)
                self.write_line("Context compacted.", "dim")
            finally:
                done.set()

        self.call_from_thread(apply)
        done.wait(timeout=10)
        return "Compacted. The conversation has been replaced by your summary."

    def show_effort_picker(self) -> None:
        if self.query("#effort-list"):
            return
        option_list = OptionList(
            *[Option(effort, id=effort) for effort in VALID_EFFORTS],
            id="effort-list",
        )
        self.query_one("#footer", Vertical).mount(option_list, before="#input")
        option_list.highlighted = VALID_EFFORTS.index(self.reasoning_effort)
        option_list.focus()

    def show_mtp_picker(self) -> None:
        if self.query("#mtp-list"):
            return
        option_list = OptionList(
            Option(f"yes — MTP ({format_tokens(tools.MODEL_CONTEXTS[tools.MODEL_MTP])} ctx)", id="yes"),
            Option(f"no — standard ({format_tokens(tools.MODEL_CONTEXTS[tools.MODEL_STD])} ctx)", id="no"),
            id="mtp-list",
        )
        self.query_one("#footer", Vertical).mount(option_list, before="#input")
        option_list.highlighted = 0
        option_list.focus()

    def set_model(self, mtp: bool) -> None:
        name = tools.MODEL_MTP if mtp else tools.MODEL_STD
        tools.MODEL = name
        global MODEL, MAX_CONTEXT, COMPACT_AT
        MODEL = name
        MAX_CONTEXT = tools.MODEL_CONTEXTS[name]
        COMPACT_AT = MAX_CONTEXT // 2
        self.update_token_count(self._token_baseline)
        self.write_line(f"Model set to {name}", "dim")

    def load_sessions(self) -> list:
        """Saved sessions, newest first, as (path, data)."""
        sessions = []
        for path in sorted(glob.glob(os.path.join(SESSIONS_DIR, "*.json")), reverse=True):
            try:
                with open(path) as f:
                    sessions.append((path, json.load(f)))
            except (OSError, json.JSONDecodeError):
                continue
        return sessions

    def show_resume_picker(self) -> None:
        if self.query("#resume-list"):
            return
        sessions = self.load_sessions()
        if not sessions:
            self.write_line("No saved sessions yet.", "dim")
            return
        options = []
        for path, data in sessions:
            tokens = data.get("tokens", estimate_tokens(data.get("messages", [])))
            if tokens >= MAX_CONTEXT:
                style = "red"
            elif tokens >= MAX_CONTEXT * 3 // 4:
                style = "yellow"
            else:
                style = ""
            line = Text()
            line.append(f"{data.get('updated', '?')}  ")
            line.append(f"{format_tokens(tokens)}  ", style=style or None)
            line.append(data.get("title") or "(empty)", style=style or None)
            options.append(Option(line, id=path))
        option_list = OptionList(*options, id="resume-list")
        self.query_one("#footer", Vertical).mount(option_list, before="#input")
        option_list.highlighted = 0
        option_list.focus()

    def resume_session(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        # keep writing to the resumed session's file
        self.session_path = path
        self.session_started = data.get("started", self.session_started)
        # fresh system message, keep only the conversation
        conversation = [
            m for m in data.get("messages", []) if m.get("role") != "system"
        ]
        self.messages = [self.build_system_message()] + conversation
        self.reasoning_effort = data.get("effort", self.reasoning_effort)
        tools.EFFORT = self.reasoning_effort
        self.query_one("#status", Static).update(f"Reasoning: {self.reasoning_effort}")

        self.messages.append({
            "role": "user",
            "content": (
                f"[Session note: this conversation was resumed at "
                f"{time.strftime('%Y-%m-%d %H:%M')}. Anything on disk "
                "may have changed in the meantime.]"
            ),
        })
        self.update_token_count(estimate_tokens(self.messages), estimate=True)
        self.save_session()
        self.write_line(
            f"Resumed session {data.get('started', '?')}",
            "dim",
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "effort-list":
            self.reasoning_effort = event.option.id
            tools.EFFORT = self.reasoning_effort
            self.query_one("#status", Static).update(f"Reasoning: {self.reasoning_effort}")
            event.option_list.remove()
            self.query_one("#input", MessageInput).focus()
        elif event.option_list.id == "resume-list":
            self.resume_session(event.option.id)
            event.option_list.remove()
            self.query_one("#input", MessageInput).focus()
        elif event.option_list.id == "mtp-list":
            self.set_model(event.option.id == "yes")
            event.option_list.remove()
            self.query_one("#input", MessageInput).focus()
        elif event.option_list.id == "approval-list":
            approved = event.option.id == "yes"
            event.option_list.remove()
            self.query_one("#input", MessageInput).focus()
            if self._pending_approval is not None:
                self._pending_approval["approved"] = approved
                self._pending_approval["event"].set()

    def report_tool_progress(self, message: str) -> None:
        """Progress from a slow tool, called from the worker thread."""
        self.call_from_thread(self.stream_text, f" {message}...", "dim")

    def report_tool_stream(self, text: str) -> None:
        """Text from a slow tool as it runs, called from the worker thread."""
        self.call_from_thread(self.stream_text, text, f"italic {COLOR_REASONING_TEXT}")

    def request_tool_approval(self, name: str, target: str) -> bool:
        pending = {"approved": False, "event": threading.Event()}
        self._pending_approval = pending

        def show():
            prompt = Text("\nRun ", style="yellow")
            prompt.append(name, style="bold yellow")
            prompt.append(" on ", style="yellow")
            prompt.append(target, style="bold yellow")
            prompt.append("?\n", style="yellow")
            self.flush_live()
            self._commit_to_log(prompt)
            option_list = OptionList(
                Option("Yes", id="yes"),
                Option("No", id="no"),
                id="approval-list",
            )
            self.query_one("#footer", Vertical).mount(option_list, before="#input")
            option_list.focus()

        self.call_from_thread(show)
        # timeout so a shutdown can't hang this thread, no answer = denied
        while not pending["event"].wait(timeout=0.5):
            if self._interrupt.is_set() or not self.is_running:
                pending["approved"] = False
                break
        self._pending_approval = None
        return pending["approved"]

    def write_markup(self, markup: str) -> None:
        """Write already-safe markup to the log."""
        self.flush_live()
        self._commit_to_log(markup)

    def _commit_to_log(self, renderable) -> None:
        """Takes trusted markup (str) or untrusted Text, which is never parsed."""
        self._history.append(renderable)
        self.query_one("#log", RichLog).write(renderable)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self.query_one("#display", VerticalScroll).scroll_end(animate=False)

    def on_resize(self, event) -> None:
        self._rewrap_history(event.size.width)

    def _rewrap_history(self, width: int) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        for markup in self._history:
            log.write(markup, width=width)
        self._scroll_to_end()

    def stream_text(self, text: str, style: str | None = None, markdown: bool = False) -> None:
        """Add streamed text. Finished lines go to the log, the partial line stays live."""
        if style != self._live_style and self._live_text:
            self._commit_live_line()
        self._live_style = style
        self._live_markdown = markdown

        self._live_text += text
        while "\n" in self._live_text:
            line, self._live_text = self._live_text.split("\n", 1)
            self._write_live_line(line, final=True)

        self._update_live_widget()

    def _render_line(self, text: str) -> Text:
        """Render a line with the current stream state."""
        return render_block_line(
            text,
            base_style=self._live_style or "",
            markdown=self._live_markdown,
            in_fence=self._in_fence,
        )

    def _write_live_line(self, text: str, final: bool) -> None:
        if self._live_markdown and not self._in_fence:
            is_row = text.strip().startswith("|")
            if is_row and self._table_start is None:
                self._table_start = len(self._history)
            elif not is_row and self._table_start is not None:
                self._flush_table()

        if self._live_markdown and text.lstrip().startswith("```"):
            self._commit_to_log(Text(text, style=self._live_style or ""))
            if self._in_fence:
                self._in_fence = False
                self._restyle_fence()
            else:
                self._in_fence = True
                self._fence_start = len(self._history) - 1
            return
        self._commit_to_log(self._render_line(text))

    def _flush_table(self) -> None:
        """Draw the buffered rows as a table once the block ends."""
        start = self._table_start
        self._table_start = None
        if start is None:
            return

        rows = [
            entry.plain for entry in self._history[start:]
            if isinstance(entry, Text) and "|" in entry.plain
        ]
        rows = [r for r in rows if not MD_TABLE_SEP.match(r.strip())]
        if len(rows) < 2:
            return

        header, *body = [split_table_row(r) for r in rows]
        table = Table(box=box.SIMPLE, header_style=f"bold {COLOR_MODEL}", border_style="dim")
        for cell in header:
            table.add_column(cell)
        for row in body:
            row = (row + [""] * len(header))[: len(header)]
            table.add_row(*(render_markdown_line(c, COLOR_MODEL) for c in row))

        del self._history[start:]
        self._history.append(table)
        self._rewrap_history(self.size.width)

    def _restyle_fence(self) -> None:
        """Closing fence arrived. Drop the marker lines and restyle as code."""
        if self._fence_start is None:
            return
        closing = len(self._history) - 1
        for i in range(self._fence_start + 1, closing):
            entry = self._history[i]
            if isinstance(entry, Text):
                self._history[i] = Text(entry.plain, style=COLOR_CODE)
        del self._history[closing]
        del self._history[self._fence_start]
        self._fence_start = None
        self._rewrap_history(self.size.width)

    def _commit_live_line(self) -> None:
        if self._pencil_active:
            self.stop_pencil()
        if self._live_text:
            self._write_live_line(self._live_text, final=True)
        self._live_text = ""
        self.query_one("#live", Static).update("")

    def _update_live_widget(self) -> None:
        if self._pencil_active:
            self.query_one("#live", Static).update(self._render_pencil_line())
        else:
            self.query_one("#live", Static).update(self._render_line(self._live_text))
        self._scroll_to_end()

    # pencil moves across the dots while a tool call's args stream in
    PENCIL_SPEED = 1.5  # dots per second
    PENCIL_TICK = 0.1

    def start_pencil(self) -> None:
        self._pencil_active = True
        self._pencil_pos = 0.0
        self._pencil_timer = self.set_interval(
            self.PENCIL_TICK, self._tick_pencil
        )

    def _pencil_dot_count(self) -> int:
        # count from the line so it matches the screen
        return len(self._live_text) - len(self._live_text.rstrip("."))

    def _tick_pencil(self) -> None:
        if not self._pencil_active:
            return
        self._pencil_pos += self.PENCIL_SPEED * self.PENCIL_TICK
        if self._pencil_pos >= self._pencil_dot_count():
            self._pencil_pos = 0.0
        self._update_live_widget()

    def stop_pencil(self) -> None:
        self._pencil_active = False
        if self._pencil_timer is not None:
            self._pencil_timer.stop()
            self._pencil_timer = None

    def _render_pencil_line(self) -> Text:
        # live line is "building <name> " + dots, pencil sits in the dots
        count = self._pencil_dot_count()
        dots = self._live_text[-count:] if count else ""
        head = self._live_text[: len(self._live_text) - count]
        return render_pencil_line(head, dots, int(self._pencil_pos), self._live_style or "")

    def flush_live(self) -> None:
        self._commit_live_line()
        if self._table_start is not None:
            self._flush_table()
        self._live_style = None
        self._live_markdown = False
        self._in_fence = False
        self._fence_start = None
        self._table_start = None

    def write_line(self, text: str, style: str | None = None) -> None:
        """Write one finished line of untrusted text to the log."""
        self.flush_live()
        self._commit_to_log(Text(text, style=style or ""))

    def note_interrupted(self) -> None:
        self.write_line("[interrupted]", "yellow")
        self.messages.append({"role": "user", "content": "[The user interrupted generation before this finished.]"})
        self._interrupt.clear()

    def stream_response(self) -> None:
        self._generating = True
        self._interrupt.clear()
        self.call_from_thread(self._reset_speed)
        self.call_from_thread(self.write_markup, f"[b {COLOR_MODEL}]Reso:[/b {COLOR_MODEL}] ")

        try:
            while True:
                # safe point, no half-finished tool call in the list
                if self._compact_notice_pending is not None:
                    self.messages.append(self._compact_notice_pending)
                    self._compact_notice_pending = None

                try:
                    content_text, tool_calls = self.stream_one_round()
                except Exception as e:
                    self.call_from_thread(self.write_line, f"Exception: {e}", "red")
                    return

                if self._interrupt.is_set():
                    if content_text:
                        self.messages.append({"role": "assistant", "content": content_text})
                    self.call_from_thread(self.note_interrupted)
                    return

                if not tool_calls:
                    self.messages.append({"role": "assistant", "content": content_text})
                    break

                assistant_msg = {"role": "assistant", "content": content_text, "tool_calls": []}
                for call in tool_calls:
                    try:
                        arguments = json.loads(call["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    assistant_msg["tool_calls"].append({
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": arguments},
                    })
                self.messages.append(assistant_msg)

                for call in tool_calls:
                    if self._interrupt.is_set():
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": "Skipped: interrupted by user.",
                        })
                        continue

                    try:
                        arguments = json.loads(call["arguments"] or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    args_preview = ", ".join(f"{k}={preview_value(v)}" for k, v in arguments.items())
                    target = describe_tool_target(call["name"], arguments)

                    approved = self.request_tool_approval(call["name"], target)
                    if not approved:
                        reason = (
                            "interrupted by user"
                            if self._interrupt.is_set()
                            else "The user denied this tool call. No reason was recorded."
                        )
                        self.call_from_thread(self.write_markup, "[dim]denied[/dim]")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": reason,
                        })
                        continue

                    self.call_from_thread(
                        self.stream_text, f"-> {call['name']}({args_preview})...", "dim"
                    )
                    result = call_tool(call["name"], arguments)
                    self.call_from_thread(self.stop_pencil)
                    self.call_from_thread(self.stream_text, " done\n", "dim")
                    if call["name"] == "compact":
                        # conversation was just compacted, drop the tool result and end the turn
                        return
                    result = cap_tool_output(result)
                    self.call_from_thread(self.note_chars, len(result))
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    })

                if self._interrupt.is_set():
                    self.call_from_thread(self.note_interrupted)
                    return

            self.call_from_thread(self.write_markup, "")
        finally:
            self._generating = False
            self.call_from_thread(self._dim_speed)
            if self._compacting:
                try:
                    self.call_from_thread(self.abort_compact_bar)
                except Exception:
                    # app is shutting down, nothing to draw on
                    self._compacting = False
            self.save_session()

    def stream_one_round(self):
        payload = {
            "model": MODEL,
            "messages": self.messages,
            "tools": TOOL_SCHEMAS,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.4,
            "chat_template_kwargs": {"reasoning_effort": self.reasoning_effort},
            **tools.reasoning_fields(self.reasoning_effort),
        }

        content_text = ""
        reasoning_ended = False
        has_reasoning = False
        tool_calls = {}

        response = requests.post(LLAMA_URL, json=payload, stream=True, timeout=300)
        if response.status_code != 200:
            self.call_from_thread(
                self.write_line, f"Error {response.status_code}: {response.text}", "red"
            )
            return content_text, []

        for line in response.iter_lines():
            if self._interrupt.is_set():
                response.close()
                break
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data = line_str[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                total = usage.get("total_tokens", 0)
                self.call_from_thread(self.update_token_count, total)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            reasoning_piece = delta.get("reasoning_content")
            if reasoning_piece:
                has_reasoning = True
                self.call_from_thread(
                    self.stream_text, reasoning_piece, f"italic {COLOR_REASONING_TEXT}"
                )
                self.call_from_thread(self.note_chars, len(reasoning_piece))
                self.call_from_thread(self.note_stream_chars, len(reasoning_piece))

            piece = delta.get("content")
            if piece:
                if has_reasoning and not reasoning_ended:
                    reasoning_ended = True
                content_text += piece
                self.call_from_thread(self.stream_text, piece, COLOR_MODEL, True)
                self.call_from_thread(self.note_chars, len(piece))
                self.call_from_thread(self.note_stream_chars, len(piece))

            for tc_delta in delta.get("tool_calls") or []:
                idx = tc_delta.get("index", 0)
                entry = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc_delta.get("id"):
                    entry["id"] = tc_delta["id"]
                fn = tc_delta.get("function") or {}
                if fn.get("name"):
                    entry["name"] = fn["name"]
                    # new call, new line, pencil starts over
                    self.call_from_thread(self.stop_pencil)
                    self.call_from_thread(
                        self.stream_text, f"\nbuilding {fn['name']} ", "dim"
                    )
                    if fn["name"] == "compact":
                        self.call_from_thread(self.start_compact_bar)
                if fn.get("arguments"):
                    before = len(entry["arguments"])
                    entry["arguments"] += fn["arguments"]
                    # one dot per 200 chars so big writes visibly grow
                    dots = len(entry["arguments"]) // 200 - before // 200
                    if dots:
                        if not self._pencil_active:
                            self.call_from_thread(self.start_pencil)
                        self.call_from_thread(self.stream_text, "." * dots, "dim")
                    self.call_from_thread(self.note_chars, len(fn["arguments"]))
                    self.call_from_thread(self.note_stream_chars, len(fn["arguments"]))
                    if entry["name"] == "compact":
                        self.call_from_thread(
                            self.advance_compact_bar, len(fn["arguments"])
                        )

        ordered_calls = [tool_calls[i] for i in sorted(tool_calls)]
        return content_text, ordered_calls


if __name__ == "__main__":
    Reso().run()
