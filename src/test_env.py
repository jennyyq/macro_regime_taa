import pandas as pd
import numpy as np
import sklearn
import yfinance as yf
import matplotlib
import scipy
import requests
import openpyxl

OPTIONAL = {
    "statsmodels" : "Ridge/BL model support",
    "networkx"    : "Regime transition network graph (Figure 6)",
    "fredapi"     : "Direct FRED API access (alternative to CSV download)",
}
 
print("=" * 45)
print("  Core dependencies")
print("=" * 45)
print("pandas      :", pd.__version__)
print("numpy       :", np.__version__)
print("sklearn     :", sklearn.__version__)
print("yfinance    :", yf.__version__)
print("matplotlib  :", matplotlib.__version__)
print("scipy       :", scipy.__version__)
print("requests    :", requests.__version__)
print("openpyxl    :", openpyxl.__version__)
 
print()
print("=" * 45)
print("  Optional dependencies")
print("=" * 45)
for pkg, reason in OPTIONAL.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"  OK  {pkg:<14} {ver}  ({reason})")
    except ImportError:
        print(f"  --  {pkg:<14} NOT FOUND  ({reason})")
 
print()
print("Environment is ready!")