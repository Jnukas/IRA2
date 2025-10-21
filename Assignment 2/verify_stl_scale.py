#!/usr/bin/env python3
"""
Quick script to verify STL file scales and suggest correct scale factors.
Run this before your main simulation to validate all mesh scales.
"""

from pathlib import Path
import trimesh
import numpy as np

def analyze_stl_scale(filepath: Path, expected_real_size_m: tuple):
    """
    Analyze an STL file and suggest scale factor.
    
    Args:
        filepath: Path to STL file
        expected_real_size_m: (x, y, z) expected real-world size in meters
    
    Returns:
        Suggested scale factor and analysis info
    """
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return None
    
    try:
        mesh = trimesh.load_mesh(str(filepath), process=False)
        bounds = mesh.bounds
        raw_size = bounds[1] - bounds[0]  # XYZ size in file units
        
        print(f"\n{'='*60}")
        print(f"📦 {filepath.name}")
        print(f"{'='*60}")
        print(f"Raw file dimensions: X={raw_size[0]:.2f}  Y={raw_size[1]:.2f}  Z={raw_size[2]:.2f}")
        
        # Detect likely unit based on magnitude
        avg_size = np.mean(raw_size)
        if avg_size > 100:
            likely_unit = "millimeters (mm)"
            suggested_scale = [0.001, 0.001, 0.001]
        elif avg_size > 10:
            likely_unit = "centimeters (cm)"
            suggested_scale = [0.01, 0.01, 0.01]
        else:
            likely_unit = "meters (m)"
            suggested_scale = [1.0, 1.0, 1.0]
        
        print(f"Likely unit: {likely_unit}")
        print(f"Suggested scale: {suggested_scale}")
        
        # Calculate what size would be with suggested scale
        scaled_size = raw_size * np.array(suggested_scale)
        print(f"Scaled dimensions (m): X={scaled_size[0]:.3f}  Y={scaled_size[1]:.3f}  Z={scaled_size[2]:.3f}")
        
        # Compare to expected if provided
        if expected_real_size_m:
            exp_x, exp_y, exp_z = expected_real_size_m
            print(f"\n📏 Expected real size (m): X={exp_x:.3f}  Y={exp_y:.3f}  Z={exp_z:.3f}")
            
            # Calculate required scale to match expected
            required_scale = np.array(expected_real_size_m) / raw_size
            print(f"✅ Exact scale to match: [{required_scale[0]:.6f}, {required_scale[1]:.6f}, {required_scale[2]:.6f}]")
            
            # Check if suggested scale is close
            error = np.abs(scaled_size - np.array(expected_real_size_m))
            if np.all(error < 0.1):  # Within 10cm
                print("✓ Suggested scale looks good!")
            else:
                print("⚠ Suggested scale may need adjustment")
        
        return suggested_scale
        
    except Exception as e:
        print(f"❌ Error analyzing {filepath.name}: {e}")
        return None


def main():
    assets_dir = Path(__file__).parent / "assets"
    
    print("\n" + "="*60)
    print("STL SCALE VERIFICATION TOOL")
    print("="*60)
    
    # Define expected real-world sizes (in meters) for known objects
    # Format: (x, y, z) - adjust these based on what you expect
    stl_expectations = {
        "Stove.stl": (0.6, 0.6, 0.9),           # Typical stove: 60x60x90cm
        "table.stl": (1.2, 0.8, 0.75),          # Typical table: 120x80x75cm
        "rightwayup.stl": (1.5, 0.9, 0.85),     # Work table
        "Potwithoutthelid.stl": (0.25, 0.25, 0.15),  # Pot: 25cm diameter, 15cm tall
        "jugfixed.stl": (0.12, 0.10, 0.20),     # Jug: ~12x10x20cm
        "pepper_grinder.stl": (0.05, 0.05, 0.15),  # Pepper grinder: 5x5x15cm
        "beef.stl": (0.20, 0.15, 0.05),         # Beef cut: 20x15x5cm
        "chicken.stl": (0.25, 0.20, 0.10),      # Chicken: 25x20x10cm
        "Fruit_and_Vegetables_Tray.stl": (0.40, 0.30, 0.10),  # Tray: 40x30x10cm
    }
    
    results = {}
    
    for stl_file, expected_size in stl_expectations.items():
        filepath = assets_dir / stl_file
        scale = analyze_stl_scale(filepath, expected_size)
        results[stl_file] = scale
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY - Copy these into your main script:")
    print("="*60)
    for stl_file, scale in results.items():
        if scale:
            print(f"{stl_file:30s} : {scale}")
    
    print("\n" + "="*60)
    print("💡 TIP: Run this script whenever you add new STL files!")
    print("="*60)


if __name__ == "__main__":
    main()