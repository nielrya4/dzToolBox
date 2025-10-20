import json
from utils.output import Output

class AgeSettings:
    def __init__(self, min_age: float = 0, max_age: float = 4500):
        self.min_age = min_age
        self.max_age = max_age
    
    def from_json(self, json_data):
        self.min_age = float(json_data["min_age"])
        self.max_age = float(json_data["max_age"])
    
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
        self.kde_bandwidth = float(json_data["kde_bandwidth"])
        self.matrix_function_type = json_data["matrix_function_type"]
        self.n_unmix_trials = int(json_data["n_unmix_trials"])
    
    def to_json(self):
        return {
            "kde_bandwidth": self.kde_bandwidth,
            "matrix_function_type": self.matrix_function_type,
            "n_unmix_trials": self.n_unmix_trials
        }

class GraphSettings:
    def __init__(self, stack_graphs: str = "false", legend: str = "true", 
                 font_size: float = 12, font_name: str = "ubuntu",
                 figure_width: int = 9, figure_height: int = 7, color_map: str = "jet"):
        self.stack_graphs = stack_graphs
        self.legend = legend
        self.font_size = font_size
        self.font_name = font_name
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.color_map = color_map
    
    def from_json(self, json_data):
        self.stack_graphs = json_data["stack_graphs"]
        self.legend = json_data["show_legend"]
        self.font_size = float(json_data["font_size"])
        self.font_name = json_data["font_name"]
        self.figure_width = int(json_data["figure_width"])
        self.figure_height = int(json_data["figure_height"])
        self.color_map = json_data["color_map"]
    
    def to_json(self):
        return {
            "stack_graphs": self.stack_graphs,
            "show_legend": self.legend,
            "font_size": self.font_size,
            "font_name": self.font_name,
            "figure_width": self.figure_width,
            "figure_height": self.figure_height,
            "color_map": self.color_map
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
