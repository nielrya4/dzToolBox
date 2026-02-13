#!/usr/bin/env python3
"""
Test script to demonstrate tabbed graph embedding
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import embedding

# Create some example figures
def create_sample_figure(title, color):
    """Create a simple test figure"""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.random.random()
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X axis', fontsize=12)
    ax.set_ylabel('Y axis', fontsize=12)
    ax.grid(True, alpha=0.3)
    return fig

# Create test figures
fig1 = create_sample_figure("Sample Scores", "blue")
fig2 = create_sample_figure("Factor Loadings", "red")
fig3 = create_sample_figure("Reconstruction", "green")

# Test 1: Single tab (should work like regular embed_graph)
print("Test 1: Single tab")
html_single = embedding.embed_tabbed_graphs(
    tabs=[{"name": "Single Graph", "fig": fig1}],
    output_id="test_single_123",
    project_id=1,
    download_formats=['svg', 'png', 'pdf'],
    is_grainalyzer=False
)
print(f"Single tab HTML length: {len(html_single)} chars")
print()

# Test 2: Multiple tabs
print("Test 2: Multiple tabs")
html_multi = embedding.embed_tabbed_graphs(
    tabs=[
        {"name": "Sample Scores", "fig": fig1},
        {"name": "Factor Loadings", "fig": fig2},
        {"name": "Reconstruction", "fig": fig3}
    ],
    output_id="test_multi_456",
    project_id=1,
    download_formats=['svg', 'png', 'pdf'],
    is_grainalyzer=True
)
print(f"Multi-tab HTML length: {len(html_multi)} chars")
print()

# Save HTML to file for inspection
with open('/tmp/tabbed_output_test.html', 'w') as f:
    f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tabbed Output Test</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</head>
<body>
    <div class="container mt-5">
        <h1>Single Tab Test</h1>
        <div class="mb-5">
            {html_single}
        </div>

        <h1>Multiple Tabs Test</h1>
        <div class="mb-5">
            {html_multi}
        </div>
    </div>
</body>
</html>
    """)

print("✓ Test HTML saved to: /tmp/tabbed_output_test.html")
print("  Open in browser to view the result")
print()

# Close figures
plt.close('all')

print("All tests completed successfully!")
