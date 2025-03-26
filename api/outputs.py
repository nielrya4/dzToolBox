from flask import request, jsonify, send_file
import io
from dz_lib.univariate import distributions
from dz_lib.univariate.data import Sample, Grain
import secrets
import json
from api.account import token_required

def register(app):
    @app.route('/api/outputs/distribution', methods=['POST'])
    @token_required
    def create_distribution_graph(current_user):
        try:
            request_data = request.get_json()
            if not request_data:
                print("no json data")
                return jsonify({"error": "Invalid JSON data."}), 400
            output_title = request_data.get("outputTitle", "Distribution Graph")
            output_type = request_data.get("outputType", "kde")
            sample_names = request_data.get("sampleNames", [])
            stacked = request_data.get("stacked", False)
            legend = request_data.get("legend", False)
            font_name = request_data.get("fontName", "Arial")
            font_size = int(request_data.get("fontSize", 12))
            color_map = request_data.get("colorMap", "viridis")
            x_min = float(request_data.get("xMin", 0))
            x_max = float(request_data.get("xMax", 100))
            fig_width = float(request_data.get("figWidth", 8))
            fig_height = float(request_data.get("figHeight", 6))
            kde_bandwidth = float(request_data.get("kdeBandwidth", 1.0))
            samples_json = request_data.get("samples")
            if not samples_json:
                print("missing 'samples'")
                return jsonify({"error": "Missing 'samples' in request data."}), 400
            samples_data = json.loads(samples_json)
            loaded_samples = []
            for sample_data in samples_data:
                grains = [Grain(grain["age"], grain["uncertainty"]) for grain in sample_data["grains"]]
                loaded_samples.append(Sample(sample_data["name"], grains))
            active_samples = [sample for sample in loaded_samples if sample.name in sample_names]
            adjusted_samples = []
            for sample in active_samples:
                if output_type == "kde":
                    sample.replace_grain_uncertainties(10)
                adjusted_samples.append(sample)
            if output_type == 'kde':
                distros = [distributions.kde_function(sample, bandwidth=kde_bandwidth) for sample in adjusted_samples]
            elif output_type == 'pdp':
                distros = [distributions.pdp_function(sample) for sample in adjusted_samples]
            elif output_type == 'cdf':
                distros = [distributions.cdf_function(distributions.kde_function(sample)) for sample in
                           adjusted_samples]
            else:
                print("unknown output type")
                return jsonify({"error": "Unsupported output type"}), 400
            graph_fig = distributions.distribution_graph(
                distributions=distros,
                title=output_title,
                stacked=stacked,
                legend=legend,
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                color_map=color_map,
                x_min=x_min,
                x_max=x_max,
                fig_width=fig_width,
                fig_height=fig_height
            )
            output_id = secrets.token_hex(15)
            img_io = io.BytesIO()
            graph_fig.savefig(img_io, format='svg')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/svg+xml', as_attachment=True,
                             download_name=f"distribution_{output_id}.svg")
        except Exception as e:
            print(e)
            return jsonify({"error": f"Error processing subset: {str(e)}"}), 500

    @app.route('/api/outputs/mds', methods=['POST'])
    @token_required
    def create_mds_graph(current_user):
        try:
            request_data = request.get_json()
            if not request_data:
                print("no json data")
                return jsonify({"error": "Invalid JSON data."}), 400
            output_title = request_data.get("outputTitle", "Distribution Graph")
            output_type = request_data.get("outputType", "kde")
            sample_names = request_data.get("sampleNames", [])
            stacked = request_data.get("stacked", False)
            legend = request_data.get("legend", False)
            font_name = request_data.get("fontName", "Arial")
            font_size = int(request_data.get("fontSize", 12))
            color_map = request_data.get("colorMap", "viridis")
            x_min = float(request_data.get("xMin", 0))
            x_max = float(request_data.get("xMax", 100))
            fig_width = float(request_data.get("figWidth", 8))
            fig_height = float(request_data.get("figHeight", 6))
            kde_bandwidth = float(request_data.get("kdeBandwidth", 1.0))
            samples_json = request_data.get("samples")
            if not samples_json:
                print("missing 'samples'")
                return jsonify({"error": "Missing 'samples' in request data."}), 400
            samples_data = json.loads(samples_json)
            loaded_samples = []
            for sample_data in samples_data:
                grains = [Grain(grain["age"], grain["uncertainty"]) for grain in sample_data["grains"]]
                loaded_samples.append(Sample(sample_data["name"], grains))
            active_samples = [sample for sample in loaded_samples if sample.name in sample_names]
            adjusted_samples = []
            for sample in active_samples:
                if output_type == "kde":
                    sample.replace_grain_uncertainties(10)
                adjusted_samples.append(sample)
            if output_type == 'kde':
                distros = [distributions.kde_function(sample, bandwidth=kde_bandwidth) for sample in adjusted_samples]
            elif output_type == 'pdp':
                distros = [distributions.pdp_function(sample) for sample in adjusted_samples]
            elif output_type == 'cdf':
                distros = [distributions.cdf_function(distributions.kde_function(sample)) for sample in
                           adjusted_samples]
            else:
                print("unknown output type")
                return jsonify({"error": "Unsupported output type"}), 400
            graph_fig = distributions.distribution_graph(
                distributions=distros,
                title=output_title,
                stacked=stacked,
                legend=legend,
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                color_map=color_map,
                x_min=x_min,
                x_max=x_max,
                fig_width=fig_width,
                fig_height=fig_height
            )
            output_id = secrets.token_hex(15)
            img_io = io.BytesIO()
            graph_fig.savefig(img_io, format='svg')
            img_io.seek(0)
            return send_file(img_io, mimetype='image/svg+xml', as_attachment=True,
                             download_name=f"distribution_{output_id}.svg")
        except Exception as e:
            print(e)
            return jsonify({"error": f"Error processing subset: {str(e)}"}), 500