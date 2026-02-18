import base64


class Output:
    def __init__(self, output_id, output_type, output_data):
        self.output_id = output_id
        self.output_type = output_type
        self.output_data = output_data

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
