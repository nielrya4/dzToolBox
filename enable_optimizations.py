#!/usr/bin/env python3
"""
Test and report optimization status for dzToolBox
Optimizations are auto-enabled when modules are imported
"""

import sys
from multiprocessing import cpu_count

def check_optimization_status():
    """Check and report the status of dzToolBox optimizations"""
    
    print("🔍 dzToolBox Optimization Status")
    print("=" * 40)
    
    cores = cpu_count()
    max_cores = min(cores, 12)
    print(f"System: {cores} CPU cores (using {max_cores})")
    
    # Test optimization modules
    optimizations_loaded = 0
    total_optimizations = 2
    
    try:
        from utils.monte_carlo_optimized import optimize_numpy_threads
        print("✅ Monte Carlo optimization: Available")
        optimizations_loaded += 1
    except ImportError as e:
        print(f"❌ Monte Carlo optimization: {e}")
    
    try:
        from utils.math_optimizations import optimize_numpy_for_performance
        print("✅ Math optimizations: Available")
        optimizations_loaded += 1
    except ImportError as e:
        print(f"❌ Math optimizations: {e}")
    
    print(f"\n📊 Optimizations loaded: {optimizations_loaded}/{total_optimizations}")
    
    if optimizations_loaded == total_optimizations:
        print("✨ All optimizations ready for high-performance computing!")
        return True
    else:
        print("⚠️  Some optimizations are missing")
        return False


if __name__ == "__main__":
    success = check_optimization_status()
    sys.exit(0 if success else 1)