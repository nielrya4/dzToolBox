import json


class Project:

    __default_settings = {
        "kde_bandwidth": 10,
        "min_age": 0,
        "max_age": 4500,
        "matrix_function_type": "kde",
        "actions_button": "true",
        "stack_graphs": "false",
        "n_trials": 10000,
        "graph_figure_settings": {
            "font_size": 12,
            "font_name": "ubuntu",
            "figure_width": 9,
            "figure_height": 7,
            "graph_color_map": "viridis"
        }
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




