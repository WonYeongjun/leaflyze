import os

for root, dirs, files in os.walk("."):
    if "__init__.py" in files:
        print(f"Found __init__.py in: {os.path.abspath(root)}")
    else:
        print(f"No __init__.py in: {os.path.abspath(root)}")
