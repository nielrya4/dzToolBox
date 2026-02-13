# Tabbed Graph Outputs

This document explains how to create tabbed graph outputs in dzToolBox, allowing multiple related graphs to be displayed in a Bootstrap tab interface.

## Overview

The `embed_tabbed_graphs()` function allows you to group multiple related graphs into a single output with tabs, similar to the original DZ Grainalyzer interface.

**Benefits:**
- Groups related visualizations together
- Cleaner output list (one output instead of many)
- Individual download buttons for each graph
- Single delete button for the entire output
- Bootstrap tab interface with smooth transitions

## Basic Usage

### Simple Example

```python
from utils import embedding
import matplotlib.pyplot as plt

# Create your figures
fig1 = create_plot_1()
fig2 = create_plot_2()
fig3 = create_plot_3()

# Create tabbed output
output_id = secrets.token_hex(15)
output_data = embedding.embed_tabbed_graphs(
    tabs=[
        {"name": "Sample Scores", "fig": fig1},
        {"name": "Factor Loadings", "fig": fig2},
        {"name": "Reconstruction", "fig": fig3}
    ],
    output_id=output_id,
    project_id=project_id,
    fig_type="matplotlib",
    img_format='svg',
    download_formats=['svg', 'png'],
    is_grainalyzer=True
)

# Add to outputs
pending_outputs.append({
    "output_id": output_id,
    "output_type": "tabbed_graph",
    "output_data": output_data
})
```

## Parameters

### `embed_tabbed_graphs()`

**Parameters:**
- `tabs` (List[Dict]): List of tab dictionaries, each with:
  - `"name"` (str): Tab label displayed to user
  - `"fig"` (matplotlib.figure.Figure): The matplotlib figure to display
- `output_id` (str): Unique identifier for this output
- `project_id` (int): Project ID
- `fig_type` (str): Figure type, default `"matplotlib"`
- `img_format` (str): Display format, default `'svg'`
- `download_formats` (List[str]): Formats available for download, default `['svg']`
- `is_grainalyzer` (bool): If True, uses grainalyzer endpoints and containers

**Returns:**
- `str`: HTML string with Bootstrap tab interface

## Real-World Example: Source Attribution

Here's how to convert a multi-sample visualization into a tabbed output:

### Before: Separate Outputs for Each Sample

```python
# OLD WAY: Creates one output per sample
for sample_idx, attr in enumerate(attribution_results):
    fig = create_attribution_plot_for_sample(attr, sample_idx)

    output_id = secrets.token_hex(15)
    output_data = embedding.embed_graph(
        fig=fig,
        output_id=output_id,
        project_id=project_id,
        is_grainalyzer=True
    )

    pending_outputs.append({
        "output_id": output_id,
        "output_type": "graph",
        "output_data": output_data
    })
```

### After: Single Tabbed Output

```python
# NEW WAY: Creates one output with multiple tabs
tabs = []

for sample_idx, attr in enumerate(attribution_results):
    fig = create_attribution_plot_for_sample(attr, sample_idx)
    tabs.append({
        "name": attr['sample_name'],  # Tab will be labeled with sample name
        "fig": fig
    })

# Create single tabbed output
output_id = secrets.token_hex(15)
output_data = embedding.embed_tabbed_graphs(
    tabs=tabs,
    output_id=output_id,
    project_id=project_id,
    fig_type="matplotlib",
    img_format='svg',
    download_formats=['svg', 'png'],
    is_grainalyzer=True
)

pending_outputs.append({
    "output_id": output_id,
    "output_type": "tabbed_graph",
    "output_data": output_data
})
```

## Complete Celery Task Example

```python
@celery_app.task(bind=True)
def my_analysis_task(self, project_id, sample_names, **kwargs):
    try:
        # ... load data and run analysis ...

        pending_outputs = []

        # Create tabbed output for multiple related graphs
        tabs = []

        # Tab 1: Overall view
        fig_overall = create_overall_plot(results)
        tabs.append({"name": "Overall", "fig": fig_overall})

        # Tab 2-N: Per-sample views
        for sample in samples:
            fig_sample = create_sample_plot(sample)
            tabs.append({"name": sample.name, "fig": fig_sample})

        # Embed as tabbed output
        output_id = secrets.token_hex(15)
        output_data = embedding.embed_tabbed_graphs(
            tabs=tabs,
            output_id=output_id,
            project_id=project_id,
            fig_type="matplotlib",
            img_format='svg',
            download_formats=['svg', 'png'],
            is_grainalyzer=True
        )

        pending_outputs.append({
            "output_id": output_id,
            "output_type": "tabbed_graph",
            "output_data": output_data
        })

        # Return for preview modal
        return {
            "status": "completed",
            "outputs": pending_outputs,
            "saved": False
        }

    except Exception as e:
        # ... error handling ...
```

## Features

### Tab Navigation
- Click tabs to switch between graphs
- First tab is active by default
- Smooth fade transitions between tabs

### Download Options
- Each graph has its own download links
- Download button labeled with tab name: "Download 'Sample1' As SVG"
- All download formats available for each graph
- Tab name is sanitized for use as filename

### Delete Functionality
- Single "Delete This Output" button
- Deletes the entire tabbed output (all tabs)
- Works with both regular and grainalyzer outputs

### Styling
- Uses Bootstrap 5 tabs
- Responsive design
- Consistent with dzToolBox styling
- Images auto-scale to container width

## When to Use Tabbed Graphs

**Good use cases:**
- Multiple views of the same data (e.g., per-sample plots)
- Sequential analysis steps (e.g., preprocessing → analysis → results)
- Related visualizations (e.g., scores + loadings + reconstruction)
- Comparing different methods or parameters

**When to use separate outputs:**
- Unrelated graphs
- Different types of outputs (graphs vs. tables)
- User needs to see multiple graphs simultaneously

## Testing

A test script is provided to verify the tabbed graph functionality:

```bash
python test_tabbed_output.py
```

This creates a test HTML file at `/tmp/tabbed_output_test.html` that you can open in a browser to see the tabbed interface in action.

## Technical Details

### HTML Structure

```html
<div>
  <!-- Tab navigation -->
  <ul class="nav nav-tabs" role="tablist">
    <li class="nav-item">
      <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab1">
        Tab 1
      </button>
    </li>
    <li class="nav-item">
      <button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab2">
        Tab 2
      </button>
    </li>
  </ul>

  <!-- Tab content -->
  <div class="tab-content">
    <div id="tab1" class="tab-pane fade show active">
      <img src="data:image/svg+xml;base64,..."/>
    </div>
    <div id="tab2" class="tab-pane fade">
      <img src="data:image/svg+xml;base64,..."/>
    </div>
  </div>

  <!-- Actions dropdown -->
  <form method="delete">
    <div class="dropdown">
      <a class="btn btn-secondary dropdown-toggle">Actions</a>
      <div class="dropdown-menu">
        <a href="..." download="Tab_1.svg">Download "Tab 1" As SVG</a>
        <a href="..." download="Tab_1.png">Download "Tab 1" As PNG</a>
        <a href="..." download="Tab_2.svg">Download "Tab 2" As SVG</a>
        <a href="..." download="Tab_2.png">Download "Tab 2" As PNG</a>
        <div class="dropdown-divider"></div>
        <button data-hx-post="/delete">Delete This Output</button>
      </div>
    </div>
  </form>
</div>
```

### Bootstrap Dependencies

The tabbed interface requires Bootstrap 5 JavaScript, which is already included in the dzToolBox editor template:

```html
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
```

No additional dependencies are needed.
