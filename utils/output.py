import base64


class Output:
    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data

    def generate_html_data(self):
        if self.type == 'graph':
            return f'<div>{self.data}</div>'
        elif self.type == 'matrix':
            return f'<div>{self.data}</div>'
        else:
            return '<h1>Invalid</h1>'
