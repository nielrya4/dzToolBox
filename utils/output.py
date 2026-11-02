import base64
from html import escape


class Output:
    def __init__(self, output_id, output_type, output_data, source_url=None, group_id=None):
        self.output_id = output_id
        self.output_type = output_type
        self.output_data = output_data
        # The request that generated this output, so it can be reconfigured, and the
        # id shared by every output that request produced (e.g. a table and a graph)
        self.source_url = source_url
        self.group_id = group_id

    @classmethod
    def from_dict(cls, data):
        return cls(
            output_id=data.get("output_id", ""),
            output_type=data.get("output_type", ""),
            output_data=data.get("output_data", ""),
            source_url=data.get("source_url") or None,
            group_id=data.get("group_id") or None,
        )

    def to_dict(self, output_data=None):
        return {
            "output_id": self.output_id,
            "output_type": self.output_type,
            "output_data": self.output_data if output_data is None else output_data,
            "source_url": self.source_url,
            "group_id": self.group_id,
        }

    def wrapper_html(self):
        """The output inside the container div the outputs list renders it in."""
        attrs = f'data-output-id="{escape(str(self.output_id))}"'
        if self.source_url and self.group_id:
            attrs += (f' data-group-id="{escape(self.group_id)}"'
                      f' data-source-url="{escape(self.source_url)}"')
        return f'<div class="mb-3 project-output" style="max-width: 100%;" {attrs}>{self.output_data}</div>'

    def generate_html_data(self):
        if self.output_type == 'graph':
            return self.output_data
        elif self.output_type == 'tabbed_graph':
            return self.output_data
        elif self.output_type == 'matrix':
            return f'<div>{self.output_data}</div>'
        elif self.output_type == 'tabbed_matrix':
            return f'<div>{self.output_data}</div>'
        else:
            return '<h1>Invalid</h1>'
