"""
Simple Premium Integration Status Checker
Checks if premium features are working without reading files
"""

import requests
import json

def quick_premium_check():
    """Quick check of premium integration status"""

    print("⚡ QUICK PREMIUM STATUS CHECK")
    print("="*40)

    base_url = "http://localhost:5000"

    # Step 1: Check if server is running
    try:
        response = requests.get(base_url, timeout=3)
        print("✅ Flask server is running")
    except:
        print("❌ Flask server not running - start with: python app.py")
        return

    # Step 2: Check premium info endpoint
    try:
        response = requests.get(f"{base_url}/api/premium-info", timeout=5)
        if response.status_code == 200:
            print("✅ Premium backend integrated")
            data = response.json()
            print(f"   Features available: {data.get('success', False)}")
        elif response.status_code == 404:
            print("❌ Premium backend NOT integrated")
            print("   🔧 Need to add premium code to app.py")
            return "missing"
        else:
            print(f"⚠️  Premium info endpoint status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking premium info: {e}")
        return "error"

    # Step 3: Quick prediction test
    test_data = {
        "ph": 6.5, "organic_carbon": 0.45, "nitrogen": 250,
        "phosphorus": 15, "potassium": 180, "sulphur": 12,
        "zinc": 0.8, "copper": 0.3, "iron": 5.2,
        "manganese": 2.5, "boron": 0.6,
        "state": "Punjab", "district": "Ludhiana",
        "soil_type": "alluvial", "crop_type": "wheat"
    }

    try:
        response = requests.post(
            f"{base_url}/api/premium-predict",
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=15
        )

        if response.status_code == 200:
            print("✅ Premium prediction working")
            data = response.json()
            if data.get('success') and data.get('bacterial_solutions'):
                print("✅ Bacterial solutions available")
                return "working"
            else:
                print("⚠️  Premium prediction partial success")
                return "partial"
        else:
            print(f"❌ Premium prediction failed: {response.status_code}")
            if response.status_code == 400:
                error_data = response.json()
                if 'rainfall' in str(error_data) or 'temperature' in str(error_data):
                    print("   🐛 Validation bug detected - need corrected backend")
                    return "validation_bug"
            return "failed"

    except Exception as e:
        print(f"❌ Premium prediction error: {e}")
        return "error"

if __name__ == "__main__":
    status = quick_premium_check()

    print("\n" + "="*40)
    if status == "working":
        print("🎉 SUCCESS: Premium system fully operational!")
    elif status == "missing":
        print("📝 ACTION REQUIRED: Integrate premium backend code")
    elif status == "validation_bug":
        print("🐛 ACTION REQUIRED: Update with corrected backend code")
    else:
        print("🔧 ACTION REQUIRED: Fix premium integration issues")