import json


class Project:

    __default_settings = {
        "kde_bandwidth": 10,
        "actions_button": "true",
        "graphs_stacked": "false",
        "n_trials": 10000
    }

    def __init__(self, name, data, outputs, settings=__default_settings):
        self.name = name
        self.data = data
        self.outputs = outputs
        self.settings = settings

    def generate_json_string(self):
        json_data = {
            "project_name": self.name,
            "data": self.data,
            "outputs": [],
            "default_settings": self.settings
        }
        for output in self.outputs:
            json_data["outputs"].append({
                "output_id": output.id,
                "output_type": output.type,
                "output_data": output.generate_html_data()
            })
        json_string = json.dumps(json_data, indent=4)
        return json_string




