from utils.sample import Sample, Grain


class Project:
    def __init__(self, name, samples, outputs):
        self.name = name
        self.samples = samples
        self.outputs = outputs

    def generate_xml_data(self):
        xml_data = f"<project name='{self.name}'>\n"
        xml_data += "<samples>\n"
        for sample in self.samples:
            xml_data += f"<sample name='{sample.name}'>\n"
            for grain in sample.grains:
                xml_data += f"<grain age='{grain.age}' uncertainty='{grain.uncertainty}'></grain>\n"
            xml_data += "</sample>\n"
        xml_data += "</samples>\n"
        xml_data += "<outputs>\n"
        for output in self.outputs:
            xml_data += (f"<output name='{output.name}' "
                         f"type='{output.output_type}' ")
            xml_data += f"{output.generate_xml_data()}\n</output>\n"
        xml_data += "</outputs>\n"
        xml_data += "</project>\n"
        return xml_data

    def add_sample(self, sample):
        self.samples.append(sample)

    def add_output(self, output):
        self.outputs.append(output)

    def get_sample_by_name(self, name):
        for sample in self.samples:
            if sample.name == name:
                return sample

    def get_output_by_name(self, name):
        for output in self.outputs:
            if output.name == name:
                return output

    def get_sample(self, i):
        return self.samples[i]

    def get_output(self, i):
        return self.outputs[i]

    def delete_sample(self, sample):
        self.samples.remove(sample)

    def delete_output(self, output):
        self.outputs.remove(output)