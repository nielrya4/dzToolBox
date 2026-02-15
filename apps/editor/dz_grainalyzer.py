"""
DZ Grainalyzer - Multivariate Tensor Factorization Analysis
"""

from flask import render_template, request, jsonify, session
from flask_login import login_required, current_user
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2_fragments import render_block
from server import database
from utils import spreadsheet, compression
from utils.project import project_from_json
from utils.output import Output

try:
    from celery_app import celery_app
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    celery_app = None

# Set up Jinja2 environment for render_block
environment = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(("html", "jinja2"))
)


def __get_project(project_id):
    """Helper to get project from database"""
    if session.get("open_project", 0) == project_id:
        file = database.get_file(project_id)
        project_content = compression.decompress(file.content)
        return project_from_json(project_content)
    else:
        return None


def register(app):
    """Register DZ Grainalyzer routes"""

    # DZ Grainalyzer is now integrated as a tab in the main editor
    # No separate page route needed

    @app.route('/projects/<int:project_id>/grainalyzer/save', methods=['POST'])
    @login_required
    def save_grainalyzer_data(project_id):
        """Save DZ Grainalyzer spreadsheet data"""
        if session.get("open_project", 0) == project_id:
            try:
                import base64
                import zlib
                import json

                compressed_data = request.get_json().get('compressedData', '')
                if not compressed_data:
                    return jsonify({"success": False, "error": "No compressed data provided"})

                # Decode and decompress the data (same pattern as main save route)
                compressed_data_bytes = base64.b64decode(compressed_data)
                decompressed_data = zlib.decompress(compressed_data_bytes).decode('utf-8')
                json_data = json.loads(decompressed_data)
                data = json_data.get("data", [])

                # Validate data format
                if not isinstance(data, list) or not all(isinstance(row, list) for row in data):
                    raise ValueError("Data is not in the expected list of lists format.")

                # Update grainalyzer_data
                project = __get_project(project_id)
                project.grainalyzer_data = spreadsheet.array_to_text(data)

                # Save to database
                compressed_proj_content = compression.compress(project.to_json())
                database.write_file(project_id, compressed_proj_content)

                return jsonify({"success": True})
            except Exception as e:
                print(f"Error saving grainalyzer data: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "error": str(e)}), 500
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/grainalyzer/outputs', methods=['GET'])
    @login_required
    def get_grainalyzer_outputs(project_id):
        """Get DZ Grainalyzer outputs"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            print(f"DEBUG: Getting grainalyzer outputs for project {project_id}")
            print(f"DEBUG: grainalyzer_outputs count: {len(project.grainalyzer_outputs)}")
            print(f"DEBUG: regular outputs count: {len(project.outputs)}")
            return jsonify([{
                "output_id": output.output_id,
                "output_type": output.output_type,
                "output_data": output.output_data
            } for output in project.grainalyzer_outputs])
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/grainalyzer/outputs/clear', methods=['POST'])
    @login_required
    def clear_grainalyzer_outputs(project_id):
        """Clear all DZ Grainalyzer outputs"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            project.grainalyzer_outputs = []

            # Save to database
            compressed_proj_content = compression.compress(project.to_json())
            database.write_file(project_id, compressed_proj_content)

            return jsonify({"success": True})
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/tensor-sample-names', methods=['GET'])
    @login_required
    def get_tensor_sample_names(project_id):
        """Get sample names for multivariate tensor factorization (row-based data)"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            # Use grainalyzer_data instead of project.data
            spreadsheet_data = spreadsheet.text_to_array(project.grainalyzer_data)
            try:
                # Read as multivariate (row-based) data for tensor factorization
                samples, feature_names = spreadsheet.read_multivariate_samples(
                    spreadsheet_array=spreadsheet_data,
                    max_age=4500
                )
                sample_names = [sample.name for sample in samples]
                return jsonify({"sample_names": sample_names, "feature_names": feature_names})
            except Exception as e:
                print(f"Error reading multivariate sample names: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"success": False, "error": str(e)})
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/outputs/new/tensor-factorization', methods=['GET'])
    @login_required
    def new_tensor_factorization(project_id):
        """Start a new tensor factorization task"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                try:
                    if not CELERY_AVAILABLE:
                        return jsonify({"error": "Celery is not configured. Please set up Redis and Celery worker."}), 500

                    # Get parameters
                    output_title = request.args.get("outputTitle", "Tensor Factorization")
                    sample_names = request.args.getlist("sampleNames")
                    rank = int(request.args.get("rank", 5))
                    model_type = request.args.get("modelType", "Tucker1")
                    output_types = request.args.getlist("outputType")
                    normalization_method = request.args.get("normalizationMethod", "standardize")
                    padding_mode = request.args.get("paddingMode", "zero")

                    # Start Celery task (pass user_id for database access)
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

                    # Return task ID immediately
                    return jsonify({"job_id": task.id, "status": "started"})

                except Exception as e:
                    import traceback
                    print(f"Tensor factorization error: {e}")
                    print(traceback.format_exc())
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"outputs": "method not allowed"})
        else:
            return jsonify({"outputs": "access_denied"})

    @app.route('/projects/<int:project_id>/grainalyzer/view-data', methods=['GET'])
    @login_required
    def view_empirical_data(project_id):
        """Generate empirical KDEs without factorization"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                try:
                    if not CELERY_AVAILABLE:
                        return jsonify({"error": "Celery is not configured. Please set up Redis and Celery worker."}), 500

                    # Get parameters
                    output_title = request.args.get("outputTitle", "Empirical KDEs")
                    sample_names = request.args.getlist("sampleNames")

                    # Start Celery task for viewing data (no factorization)
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

                    # Return task ID immediately
                    return jsonify({"job_id": task.id, "status": "started"})

                except Exception as e:
                    import traceback
                    print(f"View data error: {e}")
                    print(traceback.format_exc())
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": "method not allowed"}), 405
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/grainalyzer/find-optimal-rank', methods=['GET'])
    @login_required
    def find_optimal_rank(project_id):
        """Find optimal rank by testing multiple ranks"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                try:
                    if not CELERY_AVAILABLE:
                        return jsonify({"error": "Celery is not configured. Please set up Redis and Celery worker."}), 500

                    # Get parameters
                    output_title = request.args.get("outputTitle", "Rank Selection Analysis")
                    sample_names = request.args.getlist("sampleNames")
                    min_rank = int(request.args.get("minRank", 2))
                    max_rank = int(request.args.get("maxRank", 15))
                    model_type = request.args.get("modelType", "Tucker1")
                    normalization_method = request.args.get("normalizationMethod", "standardize")
                    output_types = request.args.getlist("outputTypes")

                    # Default to both plots if none specified
                    if not output_types:
                        output_types = ["misfit_plot", "curvature_plot"]

                    # Start Celery task for rank selection
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

                    # Return task ID immediately
                    return jsonify({"job_id": task.id, "status": "started"})

                except Exception as e:
                    import traceback
                    print(f"Find optimal rank error: {e}")
                    print(traceback.format_exc())
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": "method not allowed"}), 405
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/grainalyzer/run-factorization', methods=['GET'])
    @login_required
    def run_factorization(project_id):
        """Run tensor factorization with specified rank and outputs"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            if request.method == "GET":
                try:
                    if not CELERY_AVAILABLE:
                        return jsonify({"error": "Celery is not configured. Please set up Redis and Celery worker."}), 500

                    # Get parameters
                    output_title = request.args.get("outputTitle", "Tensor Factorization")
                    sample_names = request.args.getlist("sampleNames")
                    rank = int(request.args.get("rank", 5))
                    model_type = request.args.get("modelType", "Tucker1")
                    output_types = request.args.getlist("outputType")
                    normalization_method = request.args.get("normalizationMethod", "standardize")
                    padding_mode = request.args.get("paddingMode", "zero")

                    # Start Celery task (reuse existing tensor_factorization_task)
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

                    # Return task ID immediately
                    return jsonify({"job_id": task.id, "status": "started"})

                except Exception as e:
                    import traceback
                    print(f"Run factorization error: {e}")
                    print(traceback.format_exc())
                    return jsonify({"error": str(e)}), 500
            else:
                return jsonify({"error": "method not allowed"}), 405
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/grainalyzer/outputs/save', methods=['POST'])
    @login_required
    def save_grainalyzer_outputs(project_id):
        """Save grainalyzer outputs from preview modal"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)
            outputs_data = request.get_json().get('outputs', [])

            for output_item in outputs_data:
                project.grainalyzer_outputs.append(Output(
                    output_id=output_item['output_id'],
                    output_type=output_item['output_type'],
                    output_data=output_item['output_data']
                ))

            updated_project_content = project.to_json()
            compressed_proj_content = compression.compress(updated_project_content)
            database.write_file(project_id, compressed_proj_content)

            # Return rendered HTML for outputs container
            html = ''
            if project.grainalyzer_outputs and len(project.grainalyzer_outputs) > 0:
                for output in project.grainalyzer_outputs:
                    html += f'<div class="mb-3" style="max-width: 100%;">{output.output_data}</div>'
            else:
                html = '<p class="">Run tensor factorization to see interactive results here</p>'

            return html
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/grainalyzer/outputs/delete/<string:output_id>', methods=['POST'])
    @login_required
    def delete_grainalyzer_output(project_id, output_id):
        """Delete a single grainalyzer output"""
        if session.get("open_project", 0) == project_id:
            project = __get_project(project_id)

            # Remove the output with matching output_id
            project.grainalyzer_outputs = [
                output for output in project.grainalyzer_outputs
                if output.output_id != output_id
            ]

            # Save updated project
            updated_project_content = project.to_json()
            compressed_proj_content = compression.compress(updated_project_content)
            database.write_file(project_id, compressed_proj_content)

            # Return rendered HTML for outputs container
            html = ''
            if project.grainalyzer_outputs and len(project.grainalyzer_outputs) > 0:
                for output in project.grainalyzer_outputs:
                    html += f'<div class="mb-3" style="max-width: 100%;">{output.output_data}</div>'
            else:
                html = '<p class="">Run tensor factorization to see interactive results here</p>'

            return html
        else:
            return jsonify({"error": "access_denied"}), 403

    @app.route('/projects/<int:project_id>/outputs/job-status/<job_id>', methods=['GET'])
    @login_required
    def get_job_status(project_id, job_id):
        """Get status of a Celery background job"""
        if session.get("open_project", 0) == project_id:
            if not CELERY_AVAILABLE:
                return jsonify({"error": "Celery is not configured"}), 500

            try:
                # Use our configured celery_app, not the default one
                task = AsyncResult(job_id, app=celery_app)

                response = {
                    "job_id": job_id,
                    "status": "pending",
                    "progress": 0
                }

                # Check if we can access task state
                try:
                    state = task.state
                except Exception as state_error:
                    print(f"Error accessing task state: {state_error}")
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
                            # If outputs haven't been saved yet, include them for preview modal
                            if result.get("saved") == False:
                                # Include full outputs for preview
                                response["outputs"] = result.get("outputs", [])
                                response["saved"] = False
                                response["r2"] = result.get("r2")
                                response["best_rank"] = result.get("best_rank")
                            else:
                                # Send only summary info for already-saved outputs
                                result_summary = {
                                    "status": result.get("status"),
                                    "saved": result.get("saved"),
                                    "r2": result.get("r2"),
                                    "output_count": len(result.get("outputs", []))
                                }
                                response["result"] = result_summary
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
                print(f"=" * 80)
                print(f"ERROR in job-status endpoint for job {job_id}:")
                print(f"Error: {e}")
                print(f"Error type: {type(e).__name__}")
                print("Traceback:")
                traceback.print_exc()
                print(f"=" * 80)
                return jsonify({"error": str(e), "error_type": type(e).__name__}), 500
        else:
            return jsonify({"error": "access_denied"}), 403
