from fredapi import Fred

# 🔑 Replace this with your actual key
FRED_API_KEY = "5c86a917dbe0247c98b5e9f148833004"

try:
    fred = Fred(api_key=FRED_API_KEY)
    data = fred.get_series("CPIAUCSL").tail()
    print("✅ FRED API connection successful!")
    print(data)
except Exception as e:
    print("❌ FRED API test failed:", e)
