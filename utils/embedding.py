from pandas import DataFrame
from dz_lib.utils import encode, formats, matrices
from dz_lib.utils.encode import get_mime_type, safe_filename
import secrets


def embed_graph(fig, output_id, project_id, fig_type="matplotlib", img_format='svg', download_formats=['svg'], is_grainalyzer=False):
    accepted_image_formats = ['jpg', 'jpeg', 'png', 'pdf', 'eps', 'svg']
    if not formats.check(file_format=img_format, accepted_formats=accepted_image_formats):
        raise ValueError("Image format is not supported.")
    for download_format in download_formats:
        if not formats.check(file_format=download_format, accepted_formats=accepted_image_formats):
            raise ValueError("Download format is not supported.")
    buffer = encode.fig_to_img_buffer(fig, fig_type=fig_type, img_format=img_format)
    mime_type = get_mime_type(img_format)
    img_data = encode.buffer_to_base64(buffer, mime_type=mime_type)
    img_tag = f'<img src="{img_data}" download="image.{img_format}"/>'
    download_tags = ""
    for download_format in download_formats:
        buffer = encode.fig_to_img_buffer(fig, fig_type=fig_type, img_format=download_format)
        mime_type = get_mime_type(download_format)
        download_data = encode.buffer_to_base64(buffer, mime_type)
        download_tag = f'<a class="dropdown-item" href="{download_data}" download="image.{download_format}">Download As {download_format.upper()}</a>\n'
        download_tags += download_tag

    # Use grainalyzer-specific endpoints and containers if needed
    if is_grainalyzer:
        delete_endpoint = f"/projects/{project_id}/grainalyzer/outputs/delete/{output_id}"
        target_container = "#dz_grainalyzer_outputs"
    else:
        delete_endpoint = f"/projects/{project_id}/outputs/delete/{output_id}"
        target_container = "#outputs_container"

    html = f"""
        <div>
            {img_tag}
            <form method="delete" action="{delete_endpoint}">
                <div class="dropdown show">
                    <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                        Actions
                    </a>
                    <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                        {download_tags}
                        <button class="dropdown-item" type="submit" data-hx-post="{delete_endpoint}" data-hx-target="{target_container}" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</button>
                    </div>
                </div>
            </form>
        </div>"""
    return html


def embed_tabbed_graphs(tabs, output_id, project_id, fig_type="matplotlib", img_format='svg', download_formats=['svg'], is_grainalyzer=False):
    """
    Embed multiple graphs in a Bootstrap tabbed interface.

    Args:
        tabs: List of dicts with 'name' (tab label) and 'fig' (matplotlib figure)
              Example: [{"name": "Graph 1", "fig": fig1}, {"name": "Graph 2", "fig": fig2}]
        output_id: Unique identifier for this output
        project_id: Project ID
        fig_type: Figure type (default: "matplotlib")
        img_format: Display format (default: 'svg')
        download_formats: List of formats for download options
        is_grainalyzer: Whether this is a grainalyzer output (affects delete endpoint)

    Returns:
        HTML string with tabbed interface
    """
    accepted_image_formats = ['jpg', 'jpeg', 'png', 'pdf', 'eps', 'svg']
    if not formats.check(file_format=img_format, accepted_formats=accepted_image_formats):
        raise ValueError("Image format is not supported.")
    for download_format in download_formats:
        if not formats.check(file_format=download_format, accepted_formats=accepted_image_formats):
            raise ValueError("Download format is not supported.")

    if not tabs or len(tabs) == 0:
        raise ValueError("At least one tab is required.")

    # Use grainalyzer-specific endpoints and containers if needed
    if is_grainalyzer:
        delete_endpoint = f"/projects/{project_id}/grainalyzer/outputs/delete/{output_id}"
        target_container = "#dz_grainalyzer_outputs"
    else:
        delete_endpoint = f"/projects/{project_id}/outputs/delete/{output_id}"
        target_container = "#outputs_container"

    # Generate unique IDs for tabs (prefix with 'tab_' to ensure valid CSS selectors)
    tab_ids = [f"tab_{output_id}_{i}" for i in range(len(tabs))]

    # Build tab navigation
    tab_nav_items = []
    for i, tab in enumerate(tabs):
        active_class = "active" if i == 0 else ""
        tab_nav_items.append(f'''
            <li class="nav-item" role="presentation">
                <button class="nav-link {active_class}" id="{tab_ids[i]}_nav" data-bs-toggle="tab" data-bs-target="#{tab_ids[i]}" type="button" role="tab" aria-controls="{tab_ids[i]}" aria-selected="{'true' if i == 0 else 'false'}">
                    {tab['name']}
                </button>
            </li>''')

    tab_nav_html = f'''
        <ul class="nav nav-tabs" id="tabs_{output_id}" role="tablist">
            {''.join(tab_nav_items)}
        </ul>'''

    # Build tab content panes
    tab_panes = []
    for i, tab in enumerate(tabs):
        active_class = "show active" if i == 0 else ""

        # Generate image for this tab
        buffer = encode.fig_to_img_buffer(tab['fig'], fig_type=fig_type, img_format=img_format)
        mime_type = get_mime_type(img_format)
        img_data = encode.buffer_to_base64(buffer, mime_type=mime_type)

        tab_panes.append(f'''
            <div class="tab-pane {active_class}" id="{tab_ids[i]}" role="tabpanel" aria-labelledby="{tab_ids[i]}_nav">
                <div style="width: 100%; overflow: auto;">
                    <img src="{img_data}" style="max-width: 100%; height: auto; display: block; margin: 10px 0;"/>
                </div>
            </div>''')

    tab_content_html = f'''
        <div class="tab-content" id="tabContent_{output_id}" style="padding: 15px; min-height: 400px; display: block;">
            {''.join(tab_panes)}
        </div>'''

    # Build download options for each tab
    download_tags = []
    for i, tab in enumerate(tabs):
        for download_format in download_formats:
            buffer = encode.fig_to_img_buffer(tab['fig'], fig_type=fig_type, img_format=download_format)
            mime_type = get_mime_type(download_format)
            download_data = encode.buffer_to_base64(buffer, mime_type)
            safe_tab_name = safe_filename(tab['name'])
            download_tags.append(
                f'<a class="dropdown-item" href="{download_data}" download="{safe_tab_name}.{download_format}">Download "{tab["name"]}" As {download_format.upper()}</a>\n'
            )

    download_tags_html = ''.join(download_tags)

    # Build complete HTML (without inline script - will be initialized by global handler)
    html = f"""
        <div class="tabbed-output-container" data-tabbed-output-id="{output_id}">
            {tab_nav_html}
            {tab_content_html}
            <form method="delete" action="{delete_endpoint}">
                <div class="dropdown show" style="margin-top: 10px;">
                    <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="dropdown_{output_id}" data-bs-toggle="dropdown" aria-expanded="false">
                        Actions
                    </a>
                    <div class="dropdown-menu" aria-labelledby="dropdown_{output_id}">
                        {download_tags_html}
                        <div class="dropdown-divider"></div>
                        <button class="dropdown-item" type="submit" data-hx-post="{delete_endpoint}" data-hx-target="{target_container}" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</button>
                    </div>
                </div>
            </form>
        </div>"""

    return html


def embed_matrix(dataframe: DataFrame, output_id, project_id, title:str = None, download_formats=['xlsx']):
    accepted_image_formats = ['xlsx', 'xls', 'csv']
    download_tags = ""
    for download_format in download_formats:
        if download_format not in accepted_image_formats:
            raise ValueError(f"Download format '{download_format}' is not supported.")
    for download_format in download_formats:
        if download_format == 'xlsx':
            buffer = matrices.to_xlsx(dataframe)
        elif download_format == 'xls':
            buffer = matrices.to_xls(dataframe)
        elif download_format == 'csv':
            buffer = matrices.to_csv(dataframe)
        mime_type = get_mime_type(download_format)
        download_data = encode.buffer_to_base64(buffer, mime_type)
        download_tag = f'<a class="dropdown-item" href="{download_data}" download="{safe_filename(title if title else "data")}.{download_format}">Download As {download_format.upper()}</a>\n'
        download_tags += download_tag
    html = f"""
        <div>
            {matrices.dataframe_to_html(dataframe, title=title)}
            <form method="post" action="/projects/{project_id}/outputs/{output_id}">
                <div class="dropdown show">
                    <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                        Actions
                    </a>
                    <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                        {download_tags}
                        <button class="dropdown-item" type="submit" data-hx-post="/projects/{project_id}/outputs/delete/{output_id}" data-hx-target="#outputs_container" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</button>
                    </div>
                </div>
            </form>
        </div>"""
    return html
