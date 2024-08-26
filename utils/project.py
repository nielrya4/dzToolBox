import json


class Project:
    def __init__(self, name, data, outputs):
        self.name = name
        self.data = data
        self.outputs = outputs

    def generate_json_string(self):
        json_data = {
            "project_name": self.name,
            "data": self.data,
            "outputs": []
        }
        for output in self.outputs:
            json_data["outputs"].append({
                "output_name": output.name,
                "output_type": output.type,
                "output_data": output.generate_html_data()
            })
        json_string = json.dumps(json_data, indent=4)
        return json_string

