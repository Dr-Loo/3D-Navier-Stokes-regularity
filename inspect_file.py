#!/usr/bin/env python3
import os

filename = 'sfit_dPhi_dt_output.txt'
size = os.path.getsize(filename)
print(f"File: {filename} ({size:,} bytes)")

with open(filename, 'r', encoding='utf-8') as f:
    content = f.read()

print(f"\nFirst 500 characters:")
print("-" * 50)
print(repr(content[:500]))
print("-" * 50)

print(f"\nLooking for key patterns:")
print(f"  'Derivative' appears: {content.count('Derivative')} times")
print(f"  'C_' appears: {content.count('C_')} times")
print(f"  'dΦ' appears: {content.count('dΦ')} times")
print(f"  'dPhi' appears: {content.count('dPhi')} times")
print(f"  'expanded' appears: {content.count('expanded')} times")

# Find where the actual expression starts
if 'expanded' in content:
    idx = content.find('expanded')
    print(f"\n'expanded' found at position {idx}")
    print(f"Context around 'expanded':")
    print(repr(content[max(0, idx-50):min(len(content), idx+200)]))