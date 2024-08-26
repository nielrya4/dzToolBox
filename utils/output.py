import urllib.parse


class Output:
    def __init__(self, name, type, data):
        self.name = name
        self.type = type
        self.data = data

    def generate_html_data(self):
        if self.type == 'graph':
            encoded_data = urllib.parse.quote(self.data)
            return f'<img src="data:image/svg+xml;charset=utf-8,{encoded_data}"/>'
        elif self.type == 'matrix':
            return f'<div>{self.data}</div>'
        else:
            return '<h1>Invalid</h1>'
