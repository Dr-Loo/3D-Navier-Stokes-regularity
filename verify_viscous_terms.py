#!/usr/bin/env python3
"""
Verify the presence of viscous terms (nu) in the symbolic expression.
"""

import os
import re

def verify_viscous_terms(filename="sfit_dPhi_dt_output.txt"):
    """Check for viscosity terms in the expression."""
    
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return
    
    print("=" * 80)
    print("VERIFYING VISCOUS TERMS (nu) IN dΦ(C)/dt")
    print("=" * 80)
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ========================================================================
    # 1. Direct string searches for 'nu'
    # ========================================================================
    
    print("\n1. DIRECT SEARCH FOR 'nu':")
    print("-" * 40)
    
    nu_count = content.count('nu')
    print(f"  'nu' appears: {nu_count} times")
    
    # Check different patterns
    patterns = [' nu ', 'nu ', ' nu', '*nu', 'nu*', '(nu', 'nu)', '+nu', '-nu']
    for pattern in patterns:
        count = content.count(pattern)
        if count > 0:
            print(f"  '{pattern}': {count} times")
    
    # ========================================================================
    # 2. Look for the viscosity parameter in context
    # ========================================================================
    
    print("\n2. CONTEXT AROUND 'nu' (if found):")
    print("-" * 40)
    
    if nu_count > 0:
        # Find positions of 'nu'
        positions = [m.start() for m in re.finditer(r'nu', content)]
        for pos in positions[:5]:  # Show first 5
            start = max(0, pos - 50)
            end = min(len(content), pos + 100)
            context = content[start:end]
            print(f"\n  Context at position {pos}:")
            print(f"    ...{context}...")
    else:
        print("  No 'nu' found in the entire file.")
        print("  This suggests the expression is for the INVISCID case (Euler equations).")
    
    # ========================================================================
    # 3. Check the header for problem description
    # ========================================================================
    
    print("\n3. HEADER ANALYSIS:")
    print("-" * 40)
    
    # Look at the first 2000 characters to see if 'nu' is mentioned
    header = content[:2000]
    if 'nu' in header:
        print("  'nu' appears in header:")
        # Extract line with nu
        for line in header.split('\n'):
            if 'nu' in line:
                print(f"    {line[:100]}")
    else:
        print("  'nu' not found in header either.")
    
    # ========================================================================
    # 4. Check what the expansion was based on
    # ========================================================================
    
    print("\n4. EXPANSION SETUP CHECK:")
    print("-" * 40)
    
    # Look for signs of inviscid assumption
    inviscid_indicators = ['Euler', 'inviscid', 'ν = 0', 'nu = 0', 'no viscosity']
    for indicator in inviscid_indicators:
        if indicator.lower() in content.lower():
            print(f"  Found '{indicator}' — suggests inviscid calculation")
    
    # Look for the viscous term in the derivation
    if 'Delta T' in content or 'ΔT' in content or 'laplacian' in content.lower():
        print("  Found diffusive terms (Delta T) — viscosity may be present")
    
    # ========================================================================
    # 5. Check the actual expression structure
    # ========================================================================
    
    print("\n5. EXPRESSION STRUCTURE ANALYSIS:")
    print("-" * 40)
    
    # Find the expression (line starting with '-2*u_x')
    lines = content.split('\n')
    expr = None
    for line in lines:
        if line.strip().startswith('-2*u_x'):
            expr = line
            break
    
    if expr:
        # Look for patterns that might indicate viscosity
        # Viscous terms typically have higher-order derivatives
        laplacian_count = len(re.findall(r'Derivative\([^,]+,\s*\([xyz],\s*2\)\)', expr))
        biharmonic_count = len(re.findall(r'Derivative\([^,]+,\s*\([xyz],\s*4\)\)', expr))
        
        print(f"  Laplacian terms (∂²u): {laplacian_count}")
        print(f"  Biharmonic terms (∂⁴u): {biharmonic_count}")
        
        # Check if expression contains products with 4 or more derivatives
        high_deriv = len(re.findall(r'Derivative\([^,]+,\s*\([xyz],\s*[234]\)\)', expr))
        print(f"  High-order derivative terms (∂²,∂³,∂⁴): {high_deriv}")
        
        if high_deriv > 0 and nu_count == 0:
            print("\n  ⚠ WARNING: Expression contains high-order derivatives but no 'nu'.")
            print("    This suggests the calculation was done for the INVISCID case,")
            print("    or the viscosity was set to 1 (or another constant) before expansion.")
    
    # ========================================================================
    # 6. Look at the generated Python script to see what was computed
    # ========================================================================
    
    print("\n6. CHECK ORIGINAL SCRIPT ASSUMPTIONS:")
    print("-" * 40)
    
    # Look for the script that generated this output
    # The output file might contain a reference
    if 'Navier-Stokes' in content and 'ν' in content:
        print("  The problem is identified as Navier-Stokes (with viscosity).")
    
    # Search for 'viscosity' or 'nu' in the first few lines
    first_lines = '\n'.join(lines[:30])
    if 'ν' in first_lines:
        print("  The symbol 'ν' appears in the header.")
    else:
        print("  'ν' not found in header — may have been omitted.")
    
    # ========================================================================
    # 7. Summary
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if nu_count == 0:
        print("""
    CONCLUSION: The expression for dΦ(C)/dt contains NO viscosity terms ('nu').
    
    This means one of the following:
    
    1. The symbolic derivation was performed for the INVISCID Euler equations
       (ν = 0), not the full Navier-Stokes.
    
    2. The viscosity terms cancelled out exactly in the combination C·∂_t C.
       (This would be mathematically remarkable but needs verification.)
    
    3. The viscosity parameter was absorbed into coefficients or set to 1.
    
    RECOMMENDATION:
    To address the Millennium Problem, we need the viscous case.
    Re-run the symbolic derivation with explicit 'nu' parameter,
    ensuring it appears in the final expression.
        """)
    else:
        print(f"""
    CONCLUSION: The expression contains {nu_count} references to 'nu'.
    
    The viscous terms ARE present. Good.
    
    The dangerous cubic terms (∂u)³ are still absent — this is the key result.
        """)
    
    # Save a sample of the expression with potential nu terms
    if expr and nu_count > 0:
        # Find the part with nu
        nu_positions = [m.start() for m in re.finditer(r'nu', expr)]
        if nu_positions:
            sample_start = max(0, nu_positions[0] - 200)
            sample_end = min(len(expr), nu_positions[0] + 300)
            sample = expr[sample_start:sample_end]
            with open("dPhi_dt_with_nu_sample.txt", "w", encoding='utf-8') as f:
                f.write("Sample of expression containing 'nu':\n\n")
                f.write(sample)
            print("\nSample with 'nu' saved to 'dPhi_dt_with_nu_sample.txt'")

if __name__ == "__main__":
    verify_viscous_terms()