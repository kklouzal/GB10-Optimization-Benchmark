#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

mac_re = re.compile(r'(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}')
dgx_serial_re = re.compile(r'(DGX_SERIAL_NUMBER=)"[^"]*"')
serial_label_re = re.compile(r'(Serial Number\s*:\s*).*', re.I)
serial_key_re = re.compile(r'(serial\s*[:=]\s*).*', re.I)
token_re = re.compile(r'(token|password|passwd|secret|apikey|api_key)(\s*[:=]\s*)[^\s]+', re.I)
network_context = re.compile(r'(?i)\b(ip|ipv4|address|addresses|addr|gateway|route|router|subnet|cidr|netmask|nameserver|dns|dhcp|via|src|dst|broadcast|peer|inet|inet4)\b')
version_context = re.compile(r'(?i)\b(version|release|cuda|cudnn|tensorrt|pytorch|driver|build|dgx_ota_version|ubuntu|gcc|glibc)\b')
ipv4 = re.compile(r'(?<![A-Za-z0-9_])((?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})(?![A-Za-z0-9_])')


def redact_text(text: str) -> str:
    lines = []
    for line in text.splitlines(True):
        line = mac_re.sub('<MAC>', line)
        line = dgx_serial_re.sub(r'\1"<REDACTED>"', line)
        line = serial_label_re.sub(r'\1<REDACTED>', line)
        line = serial_key_re.sub(r'\1<REDACTED>', line)
        line = token_re.sub(r'\1\2<REDACTED>', line)
        if network_context.search(line) and not version_context.search(line):
            line = ipv4.sub('<IPv4>', line)
        lines.append(line)
    return ''.join(lines)


def main() -> int:
    root = Path(sys.argv[1])
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 30 * 1024 * 1024:
                continue
            data = path.read_bytes()
            if bytes([0]) in data:
                continue
            text = data.decode('utf-8', errors='replace')
            redacted = redact_text(text)
            if redacted != text:
                path.write_text(redacted)
        except Exception:
            continue
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
