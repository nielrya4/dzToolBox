

class Project:
    def __init__(self, name, data, outputs):
        self.name = name
        self.data = data
        self.outputs = outputs

    def generate_xml_data(self):
        xml_data = f"<project name='{self.name}'>\n"
        xml_data += "<data>\n"
        xml_data += f"{self.data}\n"
        xml_data += "</data>\n"
        xml_data += "<outputs>\n"
        for output in self.outputs:
            xml_data += f"<output name='{output.name}'"
            xml_data += f"{output.generate_xml_data()}\n"
            xml_data += "</output>\n"
        xml_data += "</outputs>\n"
        xml_data += "</project>\n"
        return xml_data
