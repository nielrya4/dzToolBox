from utils.graph import Graph


class Output:
    def __init__(self, name, output_type, data):
        self.name = name
        self.output_type = output_type
        self.data = data

    def generate_html_data(self):
        if self.output_type == 'graph' or self.output_type == 'matrix':
            return f'<div>{self.data}</div>'
        else:
            return '<h1>Invalid</h1>'
