"""
Celery tasks for long-running operations
"""

from celery_app import celery_app
from utils import spreadsheet, compression, tensor_factorization, embedding
from utils.output import Output
from utils.project import project_from_json
from server import database
from dz_lib.univariate import distributions
from dz_lib.utils import data
import secrets
import os


def clean_sample_name(sample_name):
    """Clean sample names (convert floats to ints if possible)"""
    try:
        num = float(sample_name)
        if num.is_integer():
            return str(int(num))
    except ValueError:
        pass
    return str(sample_name)


@celery_app.task(bind=True)
def tensor_factorization_task(
    self,
    project_id,
    user_id,
    output_title,
    sample_names,
    rank,
    model_type,
    output_types,
    normalization_method='standardize',
    padding_mode='zero',
    font_name='Ubuntu',
    font_size=12,
    fig_width=10,
    fig_height=8,
    color_map='viridis'
):
    """
    Run multivariate tensor factorization as a Celery task

    Parameters match the route arguments but are serialized
    """
    try:
        # Import Flask app inside function to avoid circular import
        from dzToolBox import app as flask_app

        # Update task state to show progress
        self.update_state(state='PROGRESS', meta={'status': 'Loading project...'})

        # Load project from database (needs Flask app context)
        with flask_app.app_context():
            # Import CodeFile model from dzToolBox module
            import dzToolBox as APP
            # Query database directly without relying on current_user
            file = APP.CodeFile.query.filter_by(user_id=user_id, id=project_id).first_or_404()
            project_content = compression.decompress(file.content)

        project = project_from_json(project_content)

        self.update_state(state='PROGRESS', meta={'status': 'Loading multivariate samples...'})

        # Load multivariate samples from row-based data (use grainalyzer_data)
        spreadsheet_data = spreadsheet.text_to_array(project.grainalyzer_data)

        # Convert to row format if needed (assuming data might be stored in old format)
        # For now, we assume the project data is already in row format
        # If it's in column format, we'd need to handle that differently
        loaded_samples, feature_names = spreadsheet.read_multivariate_samples(
            spreadsheet_array=spreadsheet_data,
            max_age=4500
        )

        # Filter to selected samples
        active_samples = []
        for sample in loaded_samples:
            # Clean sample name
            clean_name = clean_sample_name(sample.name)
            sample.name = clean_name
            if clean_name in sample_names:
                active_samples.append(sample)

        if len(active_samples) < 2:
            raise ValueError(f"At least 2 samples required, got {len(active_samples)}")

        self.update_state(state='PROGRESS', meta={'status': 'Building tensor from raw features...'})

        # Create tensor from multivariate samples
        tensor, metadata = tensor_factorization.create_tensor_from_multivariate_samples(
            samples=active_samples,
            feature_names=feature_names,
            padding_mode=padding_mode
        )

        self.update_state(state='PROGRESS', meta={'status': 'Normalizing features...'})

        # Normalize tensor
        # IMPORTANT: MatrixTensorFactor requires nonnegative input,
        # so we override standardize to minmax if needed
        if normalization_method == 'standardize':
            print("Warning: standardize normalization creates negative values. Using minmax instead for MatrixTensorFactor compatibility.")
            normalization_method = 'minmax'

        normalized_tensor, norm_params = tensor_factorization.normalize_tensor(
            tensor=tensor,
            method=normalization_method,
            grain_counts=metadata['grain_counts']
        )

        self.update_state(state='PROGRESS', meta={'status': 'Running factorization (this may take several minutes)...'})

        # Perform factorization
        # MatrixTensorFactor always uses nonnegative constraints
        try:
            factorization_result = tensor_factorization.factorize_tensor(
                tensor=normalized_tensor,
                rank=rank,
                model=model_type,
                nonnegative=True,  # Always True for MatrixTensorFactor
                metadata=metadata
            )
        except Exception as julia_error:
            print(f"Julia factorization error: {julia_error}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Tensor factorization failed: {julia_error}") from julia_error

        # Denormalize reconstruction for visualization
        denormalized_reconstruction = tensor_factorization.denormalize_tensor(
            normalized_tensor=factorization_result['reconstruction'],
            normalization_params=norm_params
        )

        # Calculate explained variance on original scale
        r2 = tensor_factorization.explained_variance(
            tensor, denormalized_reconstruction
        )

        self.update_state(state='PROGRESS', meta={'status': 'Generating visualizations...'})

        pending_outputs = []
        sample_names_list = metadata['sample_names']

        # Debug: log which output types were requested
        print(f"DEBUG: Generating visualizations. output_types = {output_types}")

        # Generate empirical KDEs (input data visualization)
        if "empirical_kdes" in output_types:
            n_features = len(metadata['feature_names'])
            empirical_height = max(fig_height, n_features * 2)  # At least 2 inches per feature
            graph_fig = tensor_factorization.visualize_empirical_kdes(
                tensor=tensor,
                feature_names=metadata['feature_names'],
                sample_names=sample_names_list,
                grain_counts=metadata['grain_counts'],
                title=f"{output_title}\nEmpirical Kernel Density Estimates",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=empirical_height,
                color_map='tab20'
            )
            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )
            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Generate factor loadings heatmap
        if "factor_loadings" in output_types:
            graph_fig = tensor_factorization.visualize_factor_loadings(
                factors=factorization_result['factors'],
                feature_names=metadata['feature_names'],
                title=f"{output_title}\nFactor Loadings (rank={rank}, R²={r2:.3f})",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=fig_height,
                color_map='RdBu_r'
            )
            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )
            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Generate sample scores plot
        if "sample_scores" in output_types:
            graph_fig = tensor_factorization.visualize_sample_scores(
                factors=factorization_result['factors'],
                sample_names=sample_names_list,
                title=f"{output_title} - Sample Scores (rank={rank})",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=fig_height,
                color_map=color_map
            )
            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )
            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Generate learned source KDEs
        if "learned_source_kdes" in output_types:
            n_features = len(metadata['feature_names'])
            learned_height = max(fig_height, n_features * 2)  # At least 2 inches per feature
            graph_fig = tensor_factorization.visualize_learned_source_kdes(
                reconstruction=denormalized_reconstruction,
                factors=factorization_result['factors'],
                grain_counts=metadata['grain_counts'],
                feature_names=metadata['feature_names'],
                rank=rank,
                title=f"{output_title} - Learned Source KDE Densities (rank={rank})",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=learned_height,
                color_map='Set2'
            )
            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )
            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Generate reconstruction comparison
        if "reconstruction_plot" in output_types:
            graph_fig = tensor_factorization.visualize_reconstruction_comparison(
                original=tensor,
                reconstruction=denormalized_reconstruction,
                feature_names=metadata['feature_names'],
                sample_names=sample_names_list,
                grain_counts=metadata['grain_counts'],
                sample_index=0,
                title=f"{output_title}\nReconstruction (R²={r2:.3f})",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=fig_height,
                color_map=color_map
            )
            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )
            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Generate source attribution
        if "source_attribution" in output_types:
            print(f"DEBUG: Generating source attribution visualization...")
            # Calculate grain-level source attribution
            attribution_results = tensor_factorization.calculate_source_attribution(
                original=tensor,
                factors=factorization_result['factors'],
                grain_counts=metadata['grain_counts'],
                sample_names=sample_names_list
            )

            # Visualize source attribution for each sample
            graph_fig = tensor_factorization.visualize_source_attribution(
                attributions=attribution_results,
                rank=rank,
                title=f"{output_title}\nSource Attribution (rank={rank})",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=fig_height * 1.5,  # Taller figure for multiple samples
                color_map=color_map
            )
            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )
            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })
            print(f"DEBUG: Source attribution added to pending_outputs. Total outputs: {len(pending_outputs)}")

        # Return outputs for preview (don't auto-save)
        print(f"DEBUG: Returning {len(pending_outputs)} outputs for preview")

        return {
            "status": "completed",
            "outputs": pending_outputs,  # Include full output_data for preview
            "saved": False,  # Not saved yet - will be saved via preview modal
            "r2": r2
        }

    except Exception as e:
        import traceback
        import sys

        # Print detailed error to worker logs BEFORE serialization
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print("=" * 80, file=sys.stderr)
        print(f"TENSOR FACTORIZATION ERROR:", file=sys.stderr)
        print(f"Error: {error_msg}", file=sys.stderr)
        print(f"Type: {type(e).__name__}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.stderr.flush()

        self.update_state(state='FAILURE', meta={'error': error_msg, 'traceback': error_trace})

        # Return error dict instead of raising to avoid serialization issues
        return {
            "status": "failed",
            "error": error_msg,
            "error_type": type(e).__name__
        }


@celery_app.task(bind=True)
def view_empirical_kdes_task(
    self,
    project_id,
    user_id,
    output_title,
    sample_names,
    font_name='Ubuntu',
    font_size=12,
    fig_width=10,
    fig_height=8,
    color_map='tab20'
):
    """
    View empirical KDEs without running factorization

    This is a lightweight task that just visualizes the raw input data.
    """
    try:
        from dzToolBox import app as flask_app

        self.update_state(state='PROGRESS', meta={'status': 'Loading project...'})

        # Load project from database
        with flask_app.app_context():
            import dzToolBox as APP
            file = APP.CodeFile.query.filter_by(user_id=user_id, id=project_id).first_or_404()
            project_content = compression.decompress(file.content)

        project = project_from_json(project_content)

        self.update_state(state='PROGRESS', meta={'status': 'Loading multivariate samples...'})

        # Load multivariate samples
        spreadsheet_data = spreadsheet.text_to_array(project.grainalyzer_data)
        loaded_samples, feature_names = spreadsheet.read_multivariate_samples(
            spreadsheet_array=spreadsheet_data,
            max_age=4500
        )

        # Filter to selected samples
        active_samples = []
        for sample in loaded_samples:
            clean_name = clean_sample_name(sample.name)
            sample.name = clean_name
            if clean_name in sample_names:
                active_samples.append(sample)

        if len(active_samples) < 2:
            raise ValueError(f"At least 2 samples required, got {len(active_samples)}")

        self.update_state(state='PROGRESS', meta={'status': 'Building tensor...'})

        # Create tensor from multivariate samples
        tensor, metadata = tensor_factorization.create_tensor_from_multivariate_samples(
            samples=active_samples,
            feature_names=feature_names,
            padding_mode='zero'
        )

        self.update_state(state='PROGRESS', meta={'status': 'Generating visualization...'})

        # Check if graphs should be stacked (from project settings)
        # stack_graphs = "true" means separate subplots stacked vertically
        # stack_graphs = "false" means overlaid on one plot
        stack_samples = project.settings.graph_settings.stack_graphs != "true"

        # Check if fill is enabled
        fill = project.settings.graph_settings.fill == "true"

        output_id = secrets.token_hex(15)

        # Generate empirical KDEs visualization (one figure per feature for tabs)
        # stack_samples=True means overlay, stack_samples=False means separate subplots
        feature_figures = tensor_factorization.visualize_empirical_kdes_tabbed(
            tensor=tensor,
            feature_names=metadata['feature_names'],
            sample_names=metadata['sample_names'],
            grain_counts=metadata['grain_counts'],
            title=f"{output_title}",
            font_path=f'static/global/fonts/{font_name}.ttf',
            font_size=font_size,
            fig_width=fig_width,
            fig_height=fig_height,
            color_map=color_map,
            stack_samples=stack_samples,
            fill=fill
        )

        # Create tabs - one per feature
        tabs = []
        for feature_name, fig in zip(metadata['feature_names'], feature_figures):
            tabs.append({
                "name": feature_name,
                "fig": fig
            })

        # Embed as tabbed output
        output_data = embedding.embed_tabbed_graphs(
            tabs=tabs,
            output_id=output_id,
            project_id=project_id,
            fig_type="matplotlib",
            img_format='svg',
            download_formats=['svg', 'png'],
            is_grainalyzer=True
        )

        pending_outputs = [{
            "output_id": output_id,
            "output_type": "tabbed_graph",
            "output_data": output_data
        }]

        # Return outputs for preview (don't auto-save)
        return {
            "status": "completed",
            "outputs": pending_outputs,  # Include full output_data for preview
            "saved": False  # Not saved yet - will be saved via preview modal
        }

    except Exception as e:
        import traceback
        import sys

        error_msg = str(e)
        error_trace = traceback.format_exc()
        print("=" * 80, file=sys.stderr)
        print(f"VIEW EMPIRICAL KDES ERROR:", file=sys.stderr)
        print(f"Error: {error_msg}", file=sys.stderr)
        print(f"Type: {type(e).__name__}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.stderr.flush()

        # Re-raise the exception to let Celery handle it properly
        raise


@celery_app.task(bind=True)
def find_optimal_rank_task(
    self,
    project_id,
    user_id,
    output_title,
    sample_names,
    min_rank,
    max_rank,
    model_type,
    output_types,
    normalization_method='standardize',
    font_name='Ubuntu',
    font_size=12,
    fig_width=10,
    fig_height=8,
    color_map='viridis'
):
    """
    Find optimal rank by testing multiple ranks

    This runs factorization at each rank in the range and generates
    the rank selection visualization (misfit vs ranks + optimal rank).
    """
    try:
        from dzToolBox import app as flask_app

        self.update_state(state='PROGRESS', meta={'status': 'Loading project...'})

        # Load project from database
        with flask_app.app_context():
            import dzToolBox as APP
            file = APP.CodeFile.query.filter_by(user_id=user_id, id=project_id).first_or_404()
            project_content = compression.decompress(file.content)

        project = project_from_json(project_content)

        self.update_state(state='PROGRESS', meta={'status': 'Loading multivariate samples...'})

        # Load multivariate samples
        spreadsheet_data = spreadsheet.text_to_array(project.grainalyzer_data)
        loaded_samples, feature_names = spreadsheet.read_multivariate_samples(
            spreadsheet_array=spreadsheet_data,
            max_age=4500
        )

        # Filter to selected samples
        active_samples = []
        for sample in loaded_samples:
            clean_name = clean_sample_name(sample.name)
            sample.name = clean_name
            if clean_name in sample_names:
                active_samples.append(sample)

        if len(active_samples) < 2:
            raise ValueError(f"At least 2 samples required, got {len(active_samples)}")

        self.update_state(state='PROGRESS', meta={'status': 'Calling Julia rank selection code...'})

        # Call original Julia code directly instead of reimplementing
        import subprocess
        import tempfile
        import json

        # Create temporary Excel file with row-based format expected by Julia
        temp_excel = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_json = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')
        temp_excel.close()
        temp_json.close()

        try:
            # Transform to column-based format expected by read_raw_data()
            # One sheet per feature, columns = samples, rows = grains
            import openpyxl
            wb = openpyxl.Workbook()

            # Find max grain count
            max_grains = max([len(sample.grains) for sample in active_samples])

            # Create one sheet per feature
            for feat_idx, feature_name in enumerate(feature_names):
                if feat_idx == 0:
                    ws = wb.active
                    ws.title = feature_name
                else:
                    ws = wb.create_sheet(title=feature_name)

                # Write data: rows = grains, columns = samples (no headers)
                for grain_idx in range(max_grains):
                    row = []
                    for sample in active_samples:
                        if grain_idx < len(sample.grains):
                            val = sample.grains[grain_idx].features.get(feature_name, None)
                            row.append(val)
                        else:
                            row.append(None)  # Pad with None for missing grains
                    ws.append(row)

            wb.save(temp_excel.name)
            print(f"Temporary Excel created (column-based format): {temp_excel.name}")

            # Get Julia executable path from juliapkg
            import juliapkg
            julia_exe = juliapkg.executable()
            julia_project = juliapkg.project()

            # Call Julia script with environment set to use juliacall packages
            julia_script = os.path.join(os.path.dirname(__file__), 'julia_scripts', 'rank_selection.jl')
            cmd = [julia_exe, '--project=' + julia_project, julia_script, temp_excel.name, str(min_rank), str(max_rank), temp_json.name]

            print(f"Running Julia command: {' '.join(cmd)}")
            print(f"Julia project: {julia_project}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            print(f"Julia stdout: {result.stdout}")
            if result.stderr:
                print(f"Julia stderr: {result.stderr}")

            if result.returncode != 0:
                raise RuntimeError(f"Julia script failed with code {result.returncode}: {result.stderr}")

            # Read JSON results
            with open(temp_json.name, 'r') as f:
                julia_results = json.load(f)

            if julia_results.get('status') == 'error':
                raise RuntimeError(f"Julia script error: {julia_results.get('error')}")

            ranks = julia_results['ranks']
            errors = julia_results['relative_errors']
            curvatures = julia_results['curvatures']
            best_rank = julia_results['best_rank']
            max_r2 = julia_results.get('r2', 0.0)

            print(f"Julia analysis complete: best rank = {best_rank}")

        finally:
            # Clean up temp files
            if os.path.exists(temp_excel.name):
                os.unlink(temp_excel.name)
            if os.path.exists(temp_json.name):
                os.unlink(temp_json.name)

        self.update_state(state='PROGRESS', meta={'status': 'Creating visualizations...'})

        # Julia already calculated curvatures and selected best rank
        print(f"Rank selection: Best rank is {best_rank} (from Julia analysis)")
        print(f"Generating outputs: {output_types}")

        pending_outputs = []

        # Generate visualization outputs based on user selection
        # Users can request either or both plots via output_types parameter

        # Generate misfit plot if requested
        if "misfit_plot" in output_types:
            graph_fig = tensor_factorization.visualize_misfit_plot(
                ranks=ranks,
                errors=errors,
                title=f"{output_title}",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=fig_height
            )

            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )

            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Generate curvature plot if requested
        if "curvature_plot" in output_types:
            graph_fig = tensor_factorization.visualize_curvature_plot(
                ranks=ranks,
                curvatures=curvatures,
                best_rank=best_rank,
                title=f"{output_title}\nOptimum Rank (R²={max_r2:.3f})",
                font_path=f'static/global/fonts/{font_name}.ttf',
                font_size=font_size,
                fig_width=fig_width,
                fig_height=fig_height
            )

            output_id = secrets.token_hex(15)
            output_data = embedding.embed_graph(
                fig=graph_fig,
                output_id=output_id,
                project_id=project_id,
                fig_type="matplotlib",
                img_format='svg',
                download_formats=['svg', 'png'],
                is_grainalyzer=True
            )

            pending_outputs.append({
                "output_id": output_id,
                "output_type": "graph",
                "output_data": output_data
            })

        # Return outputs for preview (don't auto-save)
        return {
            "status": "completed",
            "outputs": pending_outputs,  # Include full output_data for preview
            "saved": False,  # Not saved yet
            "best_rank": best_rank,
            "max_r2": max_r2
        }

    except Exception as e:
        import traceback
        import sys

        error_msg = str(e)
        error_trace = traceback.format_exc()
        print("=" * 80, file=sys.stderr)
        print(f"FIND OPTIMAL RANK ERROR:", file=sys.stderr)
        print(f"Error: {error_msg}", file=sys.stderr)
        print(f"Type: {type(e).__name__}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        sys.stderr.flush()

        # Re-raise the exception to let Celery handle it properly
        raise
