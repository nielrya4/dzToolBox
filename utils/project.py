import json
from utils.output import Output

class AgeSettings:
    def __init__(self, min_age: float = 0, max_age: float = 4500):
        self.min_age = min_age
        self.max_age = max_age

    def from_json(self, json_data):
        self.min_age = float(json_data.get("min_age", 0))
        self.max_age = float(json_data.get("max_age", 4500))
    
    def to_json(self):
        return {
            "min_age": self.min_age,
            "max_age": self.max_age
        }

class StatisticalSettings:
    def __init__(self, kde_bandwidth: float = 10, matrix_function_type: str = "kde", n_unmix_trials: int = 10000):
        self.kde_bandwidth = kde_bandwidth
        self.matrix_function_type = matrix_function_type
        self.n_unmix_trials = n_unmix_trials

    def from_json(self, json_data):
        self.kde_bandwidth = float(json_data.get("kde_bandwidth", 10))
        self.matrix_function_type = json_data.get("matrix_function_type", "kde")
        self.n_unmix_trials = int(json_data.get("n_unmix_trials", 10000))
    
    def to_json(self):
        return {
            "kde_bandwidth": self.kde_bandwidth,
            "matrix_function_type": self.matrix_function_type,
            "n_unmix_trials": self.n_unmix_trials
        }

class GraphSettings:
    def __init__(self, stack_graphs: str = "true", legend: str = "true",
                 font_size: float = 12, font_name: str = "ubuntu",
                 figure_width: int = 9, figure_height: int = 7, color_map: str = "jet",
                 modes_labeled: int = 0, fill: str = "false", render: str = "vector"):
        self.stack_graphs = stack_graphs
        self.legend = legend
        self.font_size = font_size
        self.font_name = font_name
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.color_map = color_map
        self.modes_labeled = modes_labeled
        self.fill = fill
        self.render = render

    def from_json(self, json_data):
        self.stack_graphs = json_data.get("stack_graphs", "true")
        self.legend = json_data.get("show_legend", "true")
        self.font_size = float(json_data.get("font_size", 12))
        self.font_name = json_data.get("font_name", "ubuntu")
        self.figure_width = int(json_data.get("figure_width", 9))
        self.figure_height = int(json_data.get("figure_height", 7))
        self.color_map = json_data.get("color_map", "jet")
        self.modes_labeled = int(json_data.get("modes_labeled", 0))
        self.fill = json_data.get("fill", "false")
        self.render = json_data.get("render", "vector")
    
    def to_json(self):
        return {
            "stack_graphs": self.stack_graphs,
            "show_legend": self.legend,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "figure_width": self.figure_width,
            "figure_height": self.figure_height,
            "color_map": self.color_map,
            "modes_labeled": self.modes_labeled,
            "fill": self.fill,
            "render": self.render
        }

class MappingSettings:
    def __init__(self, map_points: list = None):
        self.map_points = map_points if map_points is not None else []
    
    def from_json(self, json_data):
        self.map_points = json_data.get("map_points", [])
    
    def to_json(self):
        return {
            "map_points": self.map_points
        }

class Settings:
    def __init__(self):
        self.age_settings = AgeSettings()
        self.statistical_settings = StatisticalSettings()
        self.graph_settings = GraphSettings()
        self.mapping_settings = MappingSettings()

    def from_json(self, json_string):
        self.age_settings.from_json(json_string)
        self.statistical_settings.from_json(json_string)
        self.graph_settings.from_json(json_string)
        self.mapping_settings.from_json(json_string)

    def to_json(self):
        json_data = {}
        json_data.update(self.age_settings.to_json())
        json_data.update(self.statistical_settings.to_json())
        json_data.update(self.graph_settings.to_json())
        json_data.update(self.mapping_settings.to_json())
        return json_data

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
    name = json_data.get("name", "")
    data = json_data.get("data", "")
    settings = Settings()
    outputs = []
    for output in json_data.get("outputs", []):
        output_id = output.get("output_id", "")
        output_type = output.get("output_type", "")
        output_data = output.get("output_data", "")
        outputs.append(Output(output_id, output_type, output_data))
    project = Project(name, data, outputs, settings)
    if "settings" in json_data:
        project.settings.from_json(json_data["settings"])
    return project
