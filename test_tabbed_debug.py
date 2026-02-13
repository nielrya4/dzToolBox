#!/usr/bin/env python3
"""
Debug script to check tabbed graph HTML output
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import embedding

# Create a simple test figure
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, 'b-', linewidth=2)
ax.set_title("Test Graph - After Fix")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)

# Create tabbed output
html = embedding.embed_tabbed_graphs(
    tabs=[
        {"name": "Tab 1", "fig": fig},
        {"name": "Tab 2", "fig": fig}
    ],
    output_id="test_123",
    project_id=1,
    is_grainalyzer=False
)

# Save to file
with open('/tmp/tabbed_debug.html', 'w') as f:
    f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tabbed Debug</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</head>
<body>
    <div class="container mt-5">
        <h1>Tabbed Output Debug</h1>
        {html}
    </div>
</body>
</html>
    """)

print("✓ Debug HTML saved to: /tmp/tabbed_debug.html")
print("  Open in browser to inspect")

# Also print first 1000 chars of HTML to console
print("\nFirst 1000 chars of generated HTML:")
print(html[:1000])
print("\n... (truncated)")

# Check for img tags
import re
img_tags = re.findall(r'<img[^>]*>', html)
print(f"\nFound {len(img_tags)} img tags")
if img_tags:
    print("First img tag:")
    print(img_tags[0][:200] + "...")

plt.close('all')
