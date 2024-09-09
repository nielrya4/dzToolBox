import base64


class Output:
    def __init__(self, id, type, data):
        self.id = id
        self.type = type
        self.data = data

    def generate_html_data(self):
        if self.type == 'graph':
            return self.data
        elif self.type == 'matrix':
            return f'<div>{self.data}</div>'
        else:
            return '<h1>Invalid</h1>'
