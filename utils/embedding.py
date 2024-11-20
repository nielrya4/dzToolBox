from pandas import DataFrame
from io import BytesIO
from dz_lib.utils import encode, formats, matrices

def embed_graph(fig, output_id, project_id, fig_type="matplotlib", img_format='svg', download_formats=['svg']):
    accepted_image_formats = ['jpg', 'jpeg', 'png', 'pdf', 'eps', 'webp', 'svg']
    if not formats.check(file_format=img_format, accepted_formats=accepted_image_formats):
        raise ValueError("Image format is not supported.")
    for download_format in download_formats:
        if not formats.check(file_format=download_format, accepted_formats=accepted_image_formats):
            raise ValueError("Download format is not supported.")
    buffer = encode.fig_to_img_buffer(fig, fig_type=fig_type, img_format=img_format)
    img_data = encode.buffer_to_base64(buffer, img_format)
    img_tag = f'<img src="{img_data}" download="image.{img_format}"/>'

    download_tags = ""
    for download_format in download_formats:
        download_data = encode.buffer_to_base64(buffer, download_format)
        download_tag = f'<a class="dropdown-item" href="{download_data}" download="image.{download_format}">Download As {download_format.upper()}</a>\n'
        download_tags += download_tag
    html = f"""
        <div>
            {img_tag}
            <form method="delete" action="/projects/{project_id}/outputs/delete/{output_id}">
                <div class="dropdown show">
                    <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                        Actions
                    </a>
                    <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                        {download_tags}
                        <button class="dropdown-item" type="submit" href="#" data-hx-post="/projects/{project_id}/outputs/delete/{output_id}" data-hx-target="#outputs_container" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</a>
                    </div>
                </div>
            </form>
        </div>"""
    return html

def embed_matrix(dataframe: DataFrame, output_id, project_id, download_formats=['xlsx']):
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
        download_data = encode.buffer_to_base64(buffer.getvalue(), download_format)
        download_tag = f'<a class="dropdown-item" href="data:application/octet-stream;base64,{download_data}" download="data.{download_format}">Download As {download_format.upper()}</a>\n'
        download_tags += download_tag
    html = f"""
        <div>
            {matrices.dataframe_to_html(dataframe)}
            <form method="post" action="/projects/{project_id}/outputs/{output_id}">
                <div class="dropdown show">
                    <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                        Actions
                    </a>
                    <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                        {download_tags}
                        <button class="dropdown-item" type="submit" data-hx-post="/delete_output/{output_id}" data-hx-target="#outputs_container" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</button>
                    </div>
                </div>
            </form>
        </div>"""
    return html
