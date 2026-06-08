#!/usr/bin/env python3
import os
import re

# Find any .txt file that might contain the output
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]
print(f"Found .txt files: {txt_files}")

for filename in txt_files:
    size = os.path.getsize(filename)
    print(f"\n--- {filename} ({size:,} bytes) ---")
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if this looks like our expression
    if 'Derivative' in content and 'C_' in content:
        print(f"  ✓ This appears to be the dΦ/dt file!")
        
        # Get length
        print(f"  Total characters: {len(content):,}")
        
        # Count derivatives
        deriv_count = content.count('Derivative')
        print(f"  Derivative count: {deriv_count:,}")
        
        # Count C_ terms
        c_count = content.count('C_')
        print(f"  C_ term count: {c_count:,}")
        
        # Save first 2000 chars for inspection
        with open(f"excerpt_{filename}", 'w') as out:
            out.write(content[:2000])
        print(f"  Excerpt saved to excerpt_{filename}")
        
        break
else:
    print("\nNo file with 'Derivative' and 'C_' found.")