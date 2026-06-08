#!/usr/bin/env python3
"""
Inspect the structure of sfit_dPhi_dt_output.txt to find the expression.
"""

import os

filename = "sfit_dPhi_dt_output.txt"

if not os.path.exists(filename):
    print(f"File {filename} not found!")
    exit()

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File size: {len(content):,} characters")
print(f"First 2000 characters:")
print("-" * 60)
print(content[:2000])
print("-" * 60)

# Look for patterns
print("\nLooking for key patterns:")
patterns = ['dΦ/dt', 'dPhi', 'expanded', 'Derivative', '-2*u_x']
for p in patterns:
    count = content.count(p)
    print(f"  '{p}': {count} occurrences")

# Find the longest line (likely the expression)
lines = content.split('\n')
max_len = 0
max_line_idx = 0
for i, line in enumerate(lines):
    if len(line) > max_len:
        max_len = len(line)
        max_line_idx = i

print(f"\nLongest line: {max_len:,} characters at line {max_line_idx}")
if max_len > 100:
    print(f"Line {max_line_idx} starts with: {lines[max_line_idx][:200]}...")

# Find where the big expression starts
for i, line in enumerate(lines):
    if line.startswith('-2*u_x') or (line.startswith(' ') and 'Derivative' in line):
        print(f"\nLikely expression starts at line {i}")
        print(f"First 300 chars: {line[:300]}...")
        # Save the expression
        expr = ' '.join(lines[i:])
        with open("dPhi_dt_extracted.txt", "w", encoding='utf-8') as out:
            out.write(expr)
        print(f"Full expression saved to 'dPhi_dt_extracted.txt' ({len(expr):,} chars)")
        break