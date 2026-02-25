"""
Multivariate outputs routes - tensor factorization, job management, output CRUD
"""

from flask import request, jsonify, session
from flask_login import login_required, current_user
from server import database
from utils import compression
from utils.project import project_from_json
from utils.output import Output

try:
    from celery_app import celery_app
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None


def __get_project(project_id):
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None


def register(app):

    @app.route('/projects/<int:project_id>/multivariate/outputs', methods=['GET'])
    @login_required
    def get_grainalyzer_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            return jsonify([{
                "output_id": output.output_id,
                "output_type": output.output_type,
                "output_data": output.output_data
            } for output in project.grainalyzer_outputs])
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/outputs/clear', methods=['POST'])
    @login_required
    def clear_grainalyzer_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            project.grainalyzer_outputs = []
            compressed_proj_content = compression.compress(project.to_json())
            database.write_file(project_id, compressed_proj_content)
            return jsonify({"success": True})
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/outputs/save', methods=['POST'])
    @login_required
    def save_grainalyzer_outputs(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            outputs_data = request.get_json().get('outputs', [])

            for output_item in outputs_data:
                project.grainalyzer_outputs.append(Output(
                    output_id=output_item['output_id'],
                    output_type=output_item['output_type'],
                    output_data=output_item['output_data']
                ))

            compressed_proj_content = compression.compress(project.to_json())
            database.write_file(project_id, compressed_proj_content)

            html = ''
            if project.grainalyzer_outputs:
                for output in project.grainalyzer_outputs:
                    html += f'<div class="mb-3" style="max-width: 100%;">{output.output_data}</div>'
            else:
                html = '<p class="">Run tensor factorization to see interactive results here</p>'
            return html
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/outputs/delete/<string:output_id>', methods=['POST'])
    @login_required
    def delete_grainalyzer_output(project_id, output_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            project.grainalyzer_outputs = [
                o for o in project.grainalyzer_outputs if o.output_id != output_id
            ]
            compressed_proj_content = compression.compress(project.to_json())
            database.write_file(project_id, compressed_proj_content)

            html = ''
            if project.grainalyzer_outputs:
                for output in project.grainalyzer_outputs:
                    html += f'<div class="mb-3" style="max-width: 100%;">{output.output_data}</div>'
            else:
                html = '<p class="">Run tensor factorization to see interactive results here</p>'
            return html
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/outputs/new/heatmap', methods=['GET'])
    @login_required
    def new_heatmap(project_id):
        """Create a 2D KDE heatmap or surface plot for any two features of a sample"""
        if session.get("open_project", 0) == project_id:
            try:
                from utils import spreadsheet, embedding
                # Import only specific functions to avoid Julia initialization
                from utils.tensor_factorization import (
                    create_2d_kde_from_features,
                    visualize_2d_kde_surface,
                    visualize_2d_kde_heatmap
                )
                import numpy as np

                # Get parameters
                sample_name = request.args.get('sampleName')
                feature_x_idx = int(request.args.get('featureX'))
                feature_y_idx = int(request.args.get('featureY'))
                output_type = request.args.get('outputType', 'kde_2d_heatmap')
                output_title = request.args.get('outputTitle', f'Heatmap: {sample_name}')

                # Load project and settings
                project = __get_project(project_id)
                spreadsheet_data = spreadsheet.text_to_array(project.grainalyzer_data)

                # Read multivariate samples
                samples, feature_names = spreadsheet.read_multivariate_samples(
                    spreadsheet_array=spreadsheet_data,
                    max_age=4500
                )

                # Find the sample
                sample = None
                sample_idx = None
                for idx, s in enumerate(samples):
                    if s.name == sample_name:
                        sample = s
                        sample_idx = idx
                        break

                if sample is None:
                    return jsonify({"error": f"Sample '{sample_name}' not found"}), 404

                # Validate feature indices
                if feature_x_idx < 0 or feature_x_idx >= len(feature_names):
                    return jsonify({"error": f"Invalid X feature index: {feature_x_idx}"}), 400
                if feature_y_idx < 0 or feature_y_idx >= len(feature_names):
                    return jsonify({"error": f"Invalid Y feature index: {feature_y_idx}"}), 400

                feature_x_name = feature_names[feature_x_idx]
                feature_y_name = feature_names[feature_y_idx]

                # Extract feature data directly from the sample (avoid tensor creation to avoid Julia)
                # MultivariateGrain stores features as a dict with feature names as keys
                grains = samples[sample_idx].grains
                feature_x_data = np.array([g.features[feature_x_name] for g in grains])
                feature_y_data = np.array([g.features[feature_y_name] for g in grains])

                # Create 2D KDE
                kde_data = create_2d_kde_from_features(
                    feature_x_data,
                    feature_y_data,
                    grid_size=100
                )

                # Get graph settings
                font_name = project.settings.graph_settings.font_name
                font_path = f'static/global/fonts/{font_name}.ttf' if font_name and font_name.lower() != "default" else None

                font_size = project.settings.graph_settings.font_size
                fig_width = project.settings.graph_settings.figure_width
                fig_height = project.settings.graph_settings.figure_height

                # Generate output ID
                output_id = f"heatmap_{sample_name}_{feature_x_idx}_{feature_y_idx}"

                # Generate the appropriate plot
                if output_type == 'kde_2d_surface':
                    # 3D surface plot (Plotly) - return as interactive HTML
                    fig = visualize_2d_kde_surface(
                        kde_data=kde_data,
                        feature_x_name=feature_x_name,
                        feature_y_name=feature_y_name,
                        sample_name=sample_name,
                        title=output_title,
                        show_points=True,
                        font_path=font_path,
                        font_size=font_size,
                        fig_width=fig_width,
                        fig_height=fig_height
                    )

                    # Create interactive Plotly HTML
                    # Plotly is already loaded in editor.html, so don't re-include it
                    plotly_html = fig.to_html(include_plotlyjs=False, full_html=False)

                    # Generate a PNG snapshot for download
                    from dz_lib.utils import encode
                    buffer = encode.fig_to_img_buffer(fig, fig_type='plotly', img_format='png')
                    mime_type = encode.get_mime_type('png')
                    download_data = encode.buffer_to_base64(buffer, mime_type)

                    # Wrap with actions dropdown
                    delete_endpoint = f"/projects/{project_id}/multivariate/outputs/delete/{output_id}"
                    target_container = "#multivariate_outputs"

                    embedded = f"""
                        <div>
                            {plotly_html}
                            <form method="delete" action="{delete_endpoint}">
                                <div class="dropdown show">
                                    <a class="btn btn-secondary dropdown-toggle" href="#" role="button" id="{output_id}_dropdown" data-bs-toggle="dropdown" aria-expanded="false">
                                        Actions
                                    </a>
                                    <div class="dropdown-menu" aria-labelledby="{output_id}_dropdown">
                                        <a class="dropdown-item" href="{download_data}" download="heatmap_3d.png">Download As PNG</a>
                                        <button class="dropdown-item" type="submit" data-hx-post="{delete_endpoint}" data-hx-target="{target_container}" data-hx-swap="innerHTML" onclick="show_delete_output_spinner();">Delete This Output</button>
                                    </div>
                                </div>
                            </form>
                        </div>
                        <hr>"""

                else:  # kde_2d_heatmap
                    # 2D heatmap (matplotlib figure)
                    color_map = project.settings.graph_settings.color_map
                    fig = visualize_2d_kde_heatmap(
                        kde_data=kde_data,
                        feature_x_name=feature_x_name,
                        feature_y_name=feature_y_name,
                        sample_name=sample_name,
                        title=output_title,
                        show_points=True,
                        font_path=font_path,
                        font_size=font_size,
                        fig_width=fig_width,
                        fig_height=fig_height,
                        color_map=color_map
                    )

                    # Embed as PNG for display, with SVG and PNG download options
                    embedded = embedding.embed_graph(
                        fig=fig,
                        output_id=output_id,
                        project_id=project_id,
                        fig_type="matplotlib",
                        img_format='png',
                        download_formats=['svg', 'png'],
                        is_grainalyzer=True
                    )

                # Return for preview (wrapped in outputs array for preview modal)
                output_id = f"heatmap_{sample_name}_{feature_x_idx}_{feature_y_idx}"
                return jsonify({
                    "outputs": [{
                        "output_id": output_id,
                        "output_type": "graph",
                        "output_data": embedded
                    }]
                })

            except Exception as e:
                print(f"Error creating heatmap: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/view-data', methods=['GET'])
    @login_required
    def view_empirical_data(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            try:
                if not CELERY_AVAILABLE:
                    return jsonify({"error": "Celery is not configured."}), 500

                output_title = request.args.get("outputTitle", "Empirical KDEs")
                sample_names = request.args.getlist("sampleNames")

                from celery_tasks import view_empirical_kdes_task
                task = view_empirical_kdes_task.delay(
                    project_id=project_id,
                    user_id=current_user.id,
                    output_title=output_title,
                    sample_names=sample_names,
                    font_name=project.settings.graph_settings.font_name,
                    font_size=project.settings.graph_settings.font_size,
                    fig_width=project.settings.graph_settings.figure_width,
                    fig_height=project.settings.graph_settings.figure_height,
                    color_map=project.settings.graph_settings.color_map
                )
                return jsonify({"job_id": task.id, "status": "started"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/find-optimal-rank', methods=['GET'])
    @login_required
    def find_optimal_rank(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            try:
                if not CELERY_AVAILABLE:
                    return jsonify({"error": "Celery is not configured."}), 500

                output_title = request.args.get("outputTitle", "Rank Selection Analysis")
                sample_names = request.args.getlist("sampleNames")
                min_rank = int(request.args.get("minRank", 2))
                max_rank = int(request.args.get("maxRank", 15))
                model_type = request.args.get("modelType", "Tucker1")
                normalization_method = request.args.get("normalizationMethod", "standardize")
                output_types = request.args.getlist("outputTypes") or ["misfit_plot", "curvature_plot"]

                from celery_tasks import find_optimal_rank_task
                task = find_optimal_rank_task.delay(
                    project_id=project_id,
                    user_id=current_user.id,
                    output_title=output_title,
                    sample_names=sample_names,
                    min_rank=min_rank,
                    max_rank=max_rank,
                    model_type=model_type,
                    normalization_method=normalization_method,
                    output_types=output_types,
                    font_name=project.settings.graph_settings.font_name,
                    font_size=project.settings.graph_settings.font_size,
                    fig_width=project.settings.graph_settings.figure_width,
                    fig_height=project.settings.graph_settings.figure_height,
                    color_map=project.settings.graph_settings.color_map
                )
                return jsonify({"job_id": task.id, "status": "started"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/multivariate/run-factorization', methods=['GET'])
    @login_required
    def run_factorization(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            try:
                if not CELERY_AVAILABLE:
                    return jsonify({"error": "Celery is not configured."}), 500

                output_title = request.args.get("outputTitle", "Tensor Factorization")
                sample_names = request.args.getlist("sampleNames")
                rank = int(request.args.get("rank", 5))
                model_type = request.args.get("modelType", "Tucker1")
                output_types = request.args.getlist("outputType")
                normalization_method = request.args.get("normalizationMethod", "standardize")
                padding_mode = request.args.get("paddingMode", "zero")

                from celery_tasks import tensor_factorization_task
                task = tensor_factorization_task.delay(
                    project_id=project_id,
                    user_id=current_user.id,
                    output_title=output_title,
                    sample_names=sample_names,
                    rank=rank,
                    model_type=model_type,
                    output_types=output_types,
                    normalization_method=normalization_method,
                    padding_mode=padding_mode,
                    font_name=project.settings.graph_settings.font_name,
                    font_size=project.settings.graph_settings.font_size,
                    fig_width=project.settings.graph_settings.figure_width,
                    fig_height=project.settings.graph_settings.figure_height,
                    color_map=project.settings.graph_settings.color_map,
                    stack_graphs=project.settings.graph_settings.stack_graphs,
                    fill=project.settings.graph_settings.fill
                )
                return jsonify({"job_id": task.id, "status": "started"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/outputs/new/tensor-factorization', methods=['GET'])
    @login_required
    def new_tensor_factorization(project_id):
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            try:
                if not CELERY_AVAILABLE:
                    return jsonify({"error": "Celery is not configured."}), 500

                output_title = request.args.get("outputTitle", "Tensor Factorization")
                sample_names = request.args.getlist("sampleNames")
                rank = int(request.args.get("rank", 5))
                model_type = request.args.get("modelType", "Tucker1")
                output_types = request.args.getlist("outputType")
                normalization_method = request.args.get("normalizationMethod", "standardize")
                padding_mode = request.args.get("paddingMode", "zero")

                from celery_tasks import tensor_factorization_task
                task = tensor_factorization_task.delay(
                    project_id=project_id,
                    user_id=current_user.id,
                    output_title=output_title,
                    sample_names=sample_names,
                    rank=rank,
                    model_type=model_type,
                    output_types=output_types,
                    normalization_method=normalization_method,
                    padding_mode=padding_mode,
                    font_name=project.settings.graph_settings.font_name,
                    font_size=project.settings.graph_settings.font_size,
                    fig_width=project.settings.graph_settings.figure_width,
                    fig_height=project.settings.graph_settings.figure_height,
                    color_map=project.settings.graph_settings.color_map,
                    stack_graphs=project.settings.graph_settings.stack_graphs,
                    fill=project.settings.graph_settings.fill
                )
                return jsonify({"job_id": task.id, "status": "started"})
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/outputs/job-status/<job_id>', methods=['GET'])
    @login_required
    def get_job_status(project_id, job_id):
        if session.get("open_project", 0) == project_id:
            if not CELERY_AVAILABLE:
                return jsonify({"error": "Celery is not configured"}), 500
            try:
                task = AsyncResult(job_id, app=celery_app)
                response = {"job_id": job_id, "status": "pending", "progress": 0}

                try:
                    state = task.state
                except Exception as state_error:
                    return jsonify({"error": f"Cannot access task state: {state_error}"}), 500

                if state == 'PENDING':
                    response["status"] = "pending"
                elif state == 'STARTED':
                    response["status"] = "running"
                    response["progress"] = 10
                elif state == 'PROGRESS':
                    response["status"] = "running"
                    response["progress"] = 50
                    if task.info:
                        response["message"] = task.info.get('status', '')
                elif state == 'SUCCESS':
                    response["status"] = "completed"
                    response["progress"] = 100
                    try:
                        result = task.result
                        if isinstance(result, dict):
                            if result.get("saved") == False:
                                response["outputs"] = result.get("outputs", [])
                                response["saved"] = False
                                response["r2"] = result.get("r2")
                                response["best_rank"] = result.get("best_rank")
                            else:
                                response["result"] = {
                                    "status": result.get("status"),
                                    "saved": result.get("saved"),
                                    "r2": result.get("r2"),
                                    "output_count": len(result.get("outputs", []))
                                }
                        else:
                            response["result"] = {"status": "completed"}
                    except Exception as result_error:
                        print(f"Error processing task result: {result_error}")
                        import traceback
                        traceback.print_exc()
                        response["result"] = {"status": "completed"}
                elif task.state == 'FAILURE':
                    response["status"] = "failed"
                    response["error"] = str(task.info)
                else:
                    response["status"] = task.state.lower()

                return jsonify(response)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e), "error_type": type(e).__name__}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

