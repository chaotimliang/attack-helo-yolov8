#!/usr/bin/env python3
"""
Test Label Studio installation
"""

import sys
import subprocess

def test_labelstudio():
    """Test if Label Studio can be imported and run."""
    try:
        print("Testing Label Studio installation...")
        
        # Test import
        try:
            import label_studio
            print("✓ Label Studio imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import Label Studio: {e}")
            return False
        
        # Test if label-studio command exists
        try:
            result = subprocess.run(['label-studio', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ Label Studio version: {result.stdout.strip()}")
            else:
                print(f"✗ Label Studio command failed: {result.stderr}")
                return False
        except FileNotFoundError:
            print("✗ Label Studio command not found")
            return False
        except subprocess.TimeoutExpired:
            print("✗ Label Studio command timed out")
            return False
        
        print("✓ Label Studio is working!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing Label Studio: {e}")
        return False

if __name__ == "__main__":
    success = test_labelstudio()
    sys.exit(0 if success else 1)
