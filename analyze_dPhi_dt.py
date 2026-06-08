#!/usr/bin/env python3
"""
Analyze the structure of dΦ/dt expression from sfit_dPhi_dt_output.txt
"""

import re
import os

def analyze_file(filename="sfit_dPhi_dt_output.txt"):
    """Analyze the dΦ/dt expression file."""
    
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found!")
        return
    
    print("=" * 60)
    print("SFIT: dΦ/dt Expression Analysis")
    print("=" * 60)
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"\nFile size: {len(content):,} characters")
    
    # Find the expanded expression
    marker = "dΦ/dt Integrand (expanded):"
    if marker not in content:
        print(f"ERROR: Marker '{marker}' not found")
        return
    
    expr = content.split(marker)[1].strip()
    # Remove the separator line if present
    lines = expr.split('\n')
    expr_lines = []
    for line in lines:
        if not line.startswith('===') and not line.startswith('---'):
            expr_lines.append(line)
    expr = ' '.join(expr_lines)
    
    # Also remove any trailing separators
    if expr.startswith('==='):
        expr = expr.split('===')[1].strip()
    
    print(f"Expression length: {len(expr):,} characters")
    
    # Count derivative operators
    deriv_count = expr.count('Derivative')
    print(f"Derivative operators: {deriv_count:,}")
    
    # Count different types of terms
    print("\nTerm type counts:")
    
    # Count velocity field occurrences
    u_count = len(re.findall(r'u_[xyz]\([^)]+\)', expr))
    print(f"  Velocity field u: {u_count:,}")
    
    # Count viscosity
    nu_count = expr.count('nu')
    print(f"  Viscosity nu: {nu_count}")
    
    # Count derivative orders
    first_deriv = len(re.findall(r'Derivative\([^,]+,\s*[xyz]\)', expr))
    second_deriv = len(re.findall(r'Derivative\([^,]+,\s*\([xyz],\s*2\)\)', expr))
    third_deriv = len(re.findall(r'Derivative\([^,]+,\s*\([xyz],\s*3\)\)', expr))
    mixed_deriv = len(re.findall(r'Derivative\([^,]+,\s*\([xyz],\s*1\),\s*\([xyz],\s*1\)\)', expr))
    
    print(f"  First derivatives (∂u): {first_deriv:,}")
    print(f"  Second derivatives (∂²u): {second_deriv:,}")
    print(f"  Third derivatives (∂³u): {third_deriv:,}")
    print(f"  Mixed partials (∂²u): {mixed_deriv:,}")
    
    # Look for dangerous patterns
    print("\n" + "=" * 60)
    print("DANGEROUS TERM ANALYSIS")
    print("=" * 60)
    
    # Pattern: (∂u)³ type terms (potential blow-up)
    # Look for products of three first derivatives
    pattern_cubic = r'Derivative\([^)]+\)\s*\*\s*Derivative\([^)]+\)\s*\*\s*Derivative\([^)]+\)'
    cubic_matches = re.findall(pattern_cubic, expr)
    print(f"\nCubic derivative products (∂u·∂u·∂u): {len(cubic_matches):,}")
    
    # Pattern: ∂u · ∂²u (enstrophy production term)
    pattern_prod = r'Derivative\([^)]+\)\s*\*\s*Derivative\([^)]+,\s*\([xyz],\s*2\)\)'
    prod_matches = re.findall(pattern_prod, expr)
    print(f"First × second derivative products (∂u·∂²u): {len(prod_matches):,}")
    
    # Pattern: ∂³u terms (highest order)
    pattern_third = r'Derivative\([^)]+,\s*\([xyz],\s*3\)\)'
    third_matches = re.findall(pattern_third, expr)
    print(f"Third derivative terms (∂³u): {len(third_matches):,}")
    
    # Check for sign patterns (coercivity indicator)
    # Look at the first few terms
    first_500 = expr[:500]
    print(f"\nFirst 500 characters of expression:")
    print("-" * 40)
    print(first_500)
    print("-" * 40)
    
    # Check if expression starts with negative sign (good for dissipation)
    if expr.strip().startswith('-'):
        print("\n✓ Expression starts with negative sign → suggests dissipative structure")
    else:
        print("\n? Expression does not start with negative sign")
    
    # Count total terms (approximate by splitting on + and - outside parentheses)
    depth = 0
    term_count = 1
    for i, c in enumerate(expr):
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif depth == 0 and c in ('+', '-'):
            term_count += 1
    
    print(f"\nTotal approximate terms: {term_count:,}")
    
    # Save samples for manual inspection
    print("\n" + "=" * 60)
    print("SAVING SAMPLES")
    print("=" * 60)
    
    # Save first 2000 chars
    with open("dPhi_dt_excerpt_2k.txt", "w", encoding='utf-8') as f:
        f.write(expr[:2000])
    print("  Saved: dPhi_dt_excerpt_2k.txt")
    
    # Save a few sample dangerous terms if found
    if cubic_matches:
        with open("dPhi_dt_cubic_terms.txt", "w", encoding='utf-8') as f:
            for i, match in enumerate(cubic_matches[:20]):
                f.write(f"{i+1}. {match}\n\n")
        print("  Saved: dPhi_dt_cubic_terms.txt (first 20 cubic patterns)")
    
    # Save a sample of the expression with simplified notation
    # Replace long function names for readability
    simplified_sample = expr[:5000]
    simplified_sample = simplified_sample.replace('Derivative(u_x(x, y, z, t),', '∂_')
    simplified_sample = simplified_sample.replace('Derivative(u_y(x, y, z, t),', '∂_')
    simplified_sample = simplified_sample.replace('Derivative(u_z(x, y, z, t),', '∂_')
    simplified_sample = simplified_sample.replace('(x, y, z, t)', '')
    simplified_sample = simplified_sample.replace('u_x', 'u₁')
    simplified_sample = simplified_sample.replace('u_y', 'u₂')
    simplified_sample = simplified_sample.replace('u_z', 'u₃')
    
    with open("dPhi_dt_simplified_sample.txt", "w", encoding='utf-8') as f:
        f.write(simplified_sample)
    print("  Saved: dPhi_dt_simplified_sample.txt (first 5000 chars, simplified notation)")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
    The expression for dΦ/dt has been successfully extracted.
    
    Key observations:
    - Total terms: ~{term_count:,}
    - Derivative operators: {deriv_count:,}
    - Cubic products (∂u)³: {len(cubic_matches):,}
    
    Next steps:
    1. Examine 'dPhi_dt_cubic_terms.txt' to see if dangerous terms cancel
    2. Check if the expression is predominantly negative (dissipative)
    3. Run numerical simulation to test if Φ(C) remains bounded
    4. Look for patterns that can be grouped into Φ(C) itself
    """)

if __name__ == "__main__":
    analyze_file()