#!/usr/bin/env python3
"""Remove the base image's nginx `vscode server` block from port 8001.

RunPod's PyTorch base image ships /etc/nginx/nginx.conf with a server block that
listens on 8001 and proxies to localhost:8000. AetherVoice's Whisper worker (the Ear)
owns 8001, so the two collide: whisper_server.py exits with
"OSError: [Errno 98] Address already in use", and the provider reports 8001 Ready
because nginx - not Whisper - is the one answering there.

This runs at image build time, so every pod starts with the port free. It is
idempotent, keeps a .bak of the original, and refuses to write a config whose braces
do not balance (a half-cut block would take nginx down with it).
"""
import re
import shutil
import sys

PATH = "/etc/nginx/nginx.conf"


def drop_listeners(text, port):
    """Remove every server block whose body listens on `port`. Returns (text, count)."""
    out = []
    index = 0
    removed = 0
    while True:
        match = re.search(r"\bserver\s*\{", text[index:])
        if not match:
            out.append(text[index:])
            break
        start = index + match.start()
        cursor = index + match.end()
        depth = 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            # Unbalanced from here on: leave the rest exactly as it is.
            out.append(text[index:])
            break
        block = text[index + match.end():cursor - 1]
        if re.search(r"\blisten\s+(?:\[[^\]]*\]:|[\d.]*:)?%d\b" % port, block):
            out.append(text[index:start])
            removed += 1
        else:
            out.append(text[index:cursor])
        index = cursor
    return "".join(out), removed


def main():
    try:
        with open(PATH, "r") as handle:
            original = handle.read()
    except FileNotFoundError:
        print("strip_nginx_8001: %s is not in this base image - nothing to do" % PATH)
        return 0

    stripped, removed = drop_listeners(original, 8001)
    if not removed:
        print("strip_nginx_8001: no server block listens on 8001 - already clean")
        return 0
    if stripped.count("{") != stripped.count("}"):
        print("strip_nginx_8001: refusing to write an unbalanced config", file=sys.stderr)
        return 1

    shutil.copyfile(PATH, PATH + ".bak")
    with open(PATH, "w") as handle:
        handle.write(stripped)
    left = len(re.findall(r"\blisten\s+(?:\[[^\]]*\]:|[\d.]*:)?8001\b", stripped))
    print("strip_nginx_8001: removed %d server block(s) on 8001, %d listener(s) left" % (removed, left))
    return 0


if __name__ == "__main__":
    sys.exit(main())