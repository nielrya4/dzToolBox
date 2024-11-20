import json
from utils.output import Output

class Settings:
    def __init__(self,
                 min_age: float=0,
                 max_age: float=4500,
                 kde_bandwidth: float=10,
                 matrix_function_type: str="kde",
                 stack_graphs: str="false",
                 legend: str="true",
                 n_unmix_trials: int=10000,
                 font_size: float=12,
                 font_name: str="ubuntu",
                 figure_width: int=9,
                 figure_height: int=7,
                 color_map: str="jet"):
        self.min_age = min_age
        self.max_age = max_age
        self.kde_bandwidth = kde_bandwidth
        self.matrix_function_type = matrix_function_type
        self.stack_graphs = stack_graphs
        self.legend = legend
        self.n_unmix_trials = n_unmix_trials
        self.font_size = font_size
        self.font_name = font_name
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.color_map = color_map

    def from_json(self, json_string):
        self.min_age = float(json_string["min_age"])
        self.max_age = float(json_string["max_age"])
        self.kde_bandwidth = float(json_string["kde_bandwidth"])
        self.matrix_function_type = json_string["matrix_function_type"]
        self.stack_graphs = json_string["stack_graphs"]
        self.legend = json_string["show_legend"]
        self.n_unmix_trials = int(json_string["n_unmix_trials"])
        self.font_size = float(json_string["font_size"])
        self.font_name = json_string["font_name"]
        self.figure_width = int(json_string["figure_width"])
        self.figure_height = int(json_string["figure_height"])
        self.color_map = json_string["color_map"]

    def to_json(self):
        json_string = {
            "min_age": self.min_age,
            "max_age": self.max_age,
            "kde_bandwidth": self.kde_bandwidth,
            "matrix_function_type": self.matrix_function_type,
            "stack_graphs": self.stack_graphs,
            "show_legend": self.legend,
            "n_unmix_trials": self.n_unmix_trials,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "figure_width": self.figure_width,
            "figure_height": self.figure_height,
            "color_map": self.color_map
        }
        return json_string

class Project:
    def __init__(self, name: str, data: str, outputs: [Output], settings: Settings=Settings()):
        self.name = name
        self.data = data
        self.outputs = outputs
        self.settings = settings

    def delete_output(self, output_id):
        for project_output in self.outputs:
            if project_output.output_id == output_id:
                self.outputs.remove(project_output)

    def get_output(self, output_id):
        for project_output in self.outputs:
            if project_output.output_id == output_id:
                return project_output
        return None

    def to_json(self):
        json_data = {
            "name": self.name,
            "data": self.data,
            "outputs": [],
            "settings": self.settings.to_json()
        }
        for output in self.outputs:
            json_data["outputs"].append({
                "output_id": output.output_id,
                "output_type": output.output_type,
                "output_data": output.generate_html_data()
            })
        json_string = json.dumps(json_data, indent=4)
        return json_string

def project_from_json(json_data):
    json_data = json.loads(json_data)
    name = json_data.get("name")
    data = json_data.get("data")
    settings = Settings()
    outputs = []
    for output in json_data["outputs"]:
        output_id = output["output_id"]
        output_type = output["output_type"]
        output_data = output["output_data"]
        outputs.append(Output(output_id, output_type, output_data))
    project = Project(name, data, outputs, settings)
    project.settings.from_json(json_data["settings"])
    return project
