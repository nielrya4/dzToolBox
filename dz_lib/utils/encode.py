from io import BytesIO
from dz_lib.utils.formats import check
import base64
import re
import unicodedata

def buffer_to_utf8(buffer: BytesIO) -> str:
    buffer.seek(0)
    return buffer.getvalue().decode("utf-8")

def buffer_to_base64(buffer: BytesIO, mime_type: str) -> str:
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_str}"

def fig_to_img_buffer(fig, fig_type="plotly", img_format="svg") -> BytesIO:
    accepted_image_formats = ['jpg', 'jpeg', 'png', 'pdf', 'eps', 'svg']
    if fig_type == "plotly":
        if check(file_format=img_format, accepted_formats=accepted_image_formats):
            buffer = BytesIO()
            img_bytes = fig.write_image(file=buffer, format=img_format)
            buffer.seek(0)
            return buffer
        else:
            raise ValueError(f"Unsupported format: {img_format}")
    elif fig_type == "matplotlib":
        if check(file_format=img_format, accepted_formats=accepted_image_formats):
            buffer = BytesIO()
            fig.savefig(buffer, format=img_format, bbox_inches="tight")
            buffer.seek(0)
            return buffer
    else:
        raise ValueError("fig_type must be either 'plotly' or 'matplotlib'")

def fig_to_html(fig, fig_type="plotly", vector=True) -> str:
    if fig_type == "plotly":
        return fig.to_html(full_html=False)
    elif fig_type == "matplotlib":
        if vector:
            img_format = "svg"
            img_buffer = fig_to_img_buffer(fig, fig_type=fig_type, img_format=img_format)
            html = f"<div>{buffer_to_utf8(img_buffer)}</div>"
        else:
            img_format = "png"
            img_buffer = fig_to_img_buffer(fig, fig_type=fig_type, img_format=img_format)
            mime_type = get_mime_type(img_format)
            html = f"<div><img src='{buffer_to_base64(img_buffer, mime_type)}'/></div>"
    else:
        raise ValueError("fig_type must be either 'plotly' or 'matplotlib'")
    return html

def get_mime_type(file_format):
    if file_format == "svg":
        return "image/svg+xml"
    elif file_format == "png":
        return "image/png"
    elif file_format == "pdf":
        return "application/pdf"
    elif file_format == "eps":
        return "application/postscript"
    elif file_format == "webp":
        return "image/webp"
    elif file_format == "jpg" or file_format == "jpeg":
        return "image/jpeg"
    elif file_format == "xls":
        return "application/vnd.ms-excel"
    elif file_format == "xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif file_format == "csv":
        return "text/csv"
    else:
        return "application/octet-stream"

def safe_filename(text: str):
    filename = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^\w\s.-]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename