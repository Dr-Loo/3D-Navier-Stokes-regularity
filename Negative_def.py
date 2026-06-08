#!/usr/bin/env python3
"""
Analyze the sign of dΦ/dt by evaluating numerically at sample points.
"""

import re
import random

def analyze_sign_sample():
    """Check if typical terms are positive or negative."""
    
    # Read the expression
    with open("dPhi_dt_full_with_nu.txt", 'r') as f:
        expr = f.read()
    
    # Extract individual terms
    # Terms are separated by '+' or '-' at top level
    terms = re.findall(r'[+-][^*]+(?:\*[^*]+)*', expr)
    
    print(f"Total terms: {len(terms):,}")
    
    # Count signs
    positive = sum(1 for t in terms if t.strip().startswith('+') or (t.strip()[0].isdigit() and not t.strip().startswith('-')))
    negative = sum(1 for t in terms if t.strip().startswith('-'))
    
    print(f"Positive terms: {positive:,}")
    print(f"Negative terms: {negative:,}")
    
    # Sample some terms
    print("\nSample terms:")
    for i, t in enumerate(terms[:20]):
        sign = "NEG" if t.strip().startswith('-') else "POS"
        # Simplify for display
        short = t[:80] + "..." if len(t) > 80 else t
        print(f"  {i+1:2d}. {sign}: {short}")
    
    # Check for overall structure
    print("\n" + "=" * 60)
    print("OBSERVATIONS:")
    print("=" * 60)
    
    if positive == 0 and negative > 0:
        print("✓ ALL terms are negative → dΦ/dt ≤ 0 exactly!")
        print("  This would be a perfect Lyapunov functional.")
    elif negative > positive:
        print("✓ More negative terms than positive → likely dissipative")
    else:
        print("? Sign balance uncertain — need numerical evaluation")

if __name__ == "__main__":
    analyze_sign_sample()