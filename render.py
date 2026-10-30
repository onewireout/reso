"""Draws Reso's replies on screen as they stream in."""

import re

from rich.align import Align
from rich.text import Text

# colors from Tide's palette
COLOR_MODEL = "#6FAFB2"
COLOR_USER = "#D4B9A8"
COLOR_REASONING = "#8C9B91"
COLOR_REASONING_TEXT = "#E8E4DF"
COLOR_CODE = "#D4AF37"
COLOR_LINK = "#6FAFB2"

# "reso" in braille dots, with Tide's gradient
VERSION = "v0.2"
BANNER = [
    " · ✦ ⣿⠞⠉⠓⢠⣞⣉⣙⣦⠀⢾⣉⣉⠁⢠⡞⠉⠙⣦",
    "     ⣿⠀⠀⠀⠘⢧⣀⣀⠤⠀⣀⣀⣀⡿⠘⢧⣀⣠⠟ ✦ ·",
    f"     ╶────────◆────────╴ {VERSION}",
]
TIDE_START = (70, 120, 123)
TIDE_END = (212, 185, 168)


def render_banner() -> Align:
    width = max(len(line) for line in BANNER)
    text = Text("\n")
    for line in BANNER:
        for x, ch in enumerate(line):
            t = x / (width - 1)
            r, g, b = (round(s + (e - s) * t) for s, e in zip(TIDE_START, TIDE_END))
            text.append(ch, style=f"rgb({r},{g},{b})")
        text.append("\n")
    # center the whole block so the lines stay lined up
    return Align.center(text)

# only match closed markers, so a half-typed **bold stays raw
# no underscores, snake_case would turn italic all the time
MD_INLINE = re.compile(
    r"(?P<code>`[^`\n]+`)"
    r"|(?P<link>\[(?P<text>[^\]\n]+)\]\((?P<url>[^)\s\n]+)\))"
    r"|(?P<bolditalic>(?:^|(?<=\s))\*\*\*(?!\s)[^*\n]*?[^\s*]\*\*\*(?![\w*]))"
    r"|(?P<bold>(?:^|(?<=\s))\*\*(?!\s)[^*\n]*?[^\s*]\*\*(?![\w*]))"
    r"|(?P<italic>(?:^|(?<=\s))\*(?!\s)[^*\n]*?[^\s*]\*(?![\w*]))"
    r"|(?P<strike>~~(?!\s)[^~\n]*?[^\s~]~~)"
)

MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
MD_NUMBERED = re.compile(r"^(\d{1,3})[.)]\s+(.*)$")
MD_QUOTE = re.compile(r"^>\s?(.*)$")
MD_RULE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")
MD_TABLE_SEP = re.compile(r"^\|?[\s:|-]*-[\s:|-]*\|?$")


def split_table_row(line: str) -> list:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def render_markdown_line(text: str, base_style: str) -> Text:
    """Inline stuff only: code, links, bold, italics."""
    out = Text(style=base_style)
    pos = 0
    for match in MD_INLINE.finditer(text):
        if match.start() > pos:
            out.append(text[pos:match.start()])
        if match.group("code"):
            out.append(match.group()[1:-1], style=COLOR_CODE)
        elif match.group("link"):
            out.append(match.group("text"), style=f"underline {COLOR_LINK}")
            out.append(f" ({match.group('url')})", style="dim")
        elif match.group("bolditalic"):
            out.append(match.group()[3:-3], style="bold italic")
        elif match.group("bold"):
            out.append(match.group()[2:-2], style="bold")
        elif match.group("italic"):
            out.append(match.group()[1:-1], style="italic")
        else:
            out.append(match.group()[2:-2], style="strike")
        pos = match.end()
    out.append(text[pos:])
    return out


def render_block_line(
    text: str, base_style: str = "", markdown: bool = False, in_fence: bool = False
) -> Text:
    """Applies headings, rules, quotes and lists to one line."""
    if not markdown or in_fence:
        return Text(text, style=base_style)

    stripped = text.lstrip()
    indent = len(text) - len(stripped)

    heading = MD_HEADING.match(stripped)
    if heading:
        out = Text(" " * indent, style=base_style)
        out.append_text(render_markdown_line(heading.group(2), f"bold {base_style}"))
        return out

    if MD_RULE.match(stripped):
        return Text("─" * 40, style="dim")

    quote = MD_QUOTE.match(stripped)
    if quote:
        out = Text(" " * indent + "│ ", style="dim")
        out.append_text(render_markdown_line(quote.group(1), f"italic {base_style}"))
        return out

    numbered = MD_NUMBERED.match(stripped)
    if numbered:
        out = Text(f"{' ' * indent}{numbered.group(1)}. ", style="bold")
        out.append_text(render_markdown_line(numbered.group(2), base_style))
        return out

    # a leading bullet isn't italics
    if stripped[:2] in ("- ", "* "):
        out = Text(" " * indent + "• ", style=base_style)
        out.append_text(render_markdown_line(stripped[2:], base_style))
        return out

    return render_markdown_line(text, base_style)


def render_pencil_line(head: str, dots: str, pos: int, base_style: str, pencil: str = "✎") -> Text:
    """The 'building <name> ...' line with the pencil in the dots."""
    pos = max(0, min(pos, len(dots)))
    text = Text()
    text.append(head, style=base_style)
    text.append(dots[:pos], style=base_style)
    text.append(pencil, style=COLOR_MODEL)
    text.append(dots[pos:], style=base_style)
    return text
