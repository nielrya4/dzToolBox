"""
Test the full tensor factorization pipeline
"""
import sys
from utils import spreadsheet, tensor_factorization
import numpy as np

print("=" * 60)
print("Testing Full Tensor Factorization Pipeline")
print("=" * 60)

try:
    # Step 1: Read data (simulating retrieval from DB)
    print("\n1. Reading and preparing data...")
    data_from_excel = spreadsheet.excel_to_array('static/global/docs/example_tensor_data.xlsx')
    transposed = np.transpose(data_from_excel)
    stored_data = np.ndarray.tolist(transposed)
    retrieved_data = spreadsheet.text_to_array(spreadsheet.array_to_text(stored_data))

    samples, feature_names = spreadsheet.read_multivariate_samples(
        spreadsheet_array=retrieved_data,
        max_age=4500
    )
    print(f"   ✓ Loaded {len(samples)} samples with {len(feature_names)} features")

    # Step 2: Select subset of samples (simulating user selection)
    print("\n2. Selecting samples...")
    sample_names_to_use = ["SINK NAME 1", "SINK NAME 2", "SINK NAME 3"]
    active_samples = [s for s in samples if s.name in sample_names_to_use]
    print(f"   ✓ Selected {len(active_samples)} samples")

    # Step 3: Create tensor
    print("\n3. Creating tensor from multivariate samples...")
    tensor, metadata = tensor_factorization.create_tensor_from_multivariate_samples(
        samples=active_samples,
        feature_names=feature_names,
        padding_mode='zero'
    )
    print(f"   ✓ Tensor shape: {tensor.shape}")
    print(f"   ✓ Metadata: {metadata.keys()}")

    # Step 4: Normalize
    print("\n4. Normalizing tensor with minmax (required for nonnegative factorization)...")
    normalized_tensor, norm_params = tensor_factorization.normalize_tensor(
        tensor=tensor,
        method='minmax',  # Use minmax for nonnegative data
        grain_counts=metadata['grain_counts']
    )
    print(f"   ✓ Normalized tensor shape: {normalized_tensor.shape}")
    print(f"   ✓ Normalization params: {norm_params.keys()}")
    print(f"   ✓ Data range: [{normalized_tensor.min():.4f}, {normalized_tensor.max():.4f}]")

    # Step 5: Factorize
    print("\n5. Running tensor factorization (THIS MAY TAKE A MINUTE)...")
    factorization_result = tensor_factorization.factorize_tensor(
        tensor=normalized_tensor,
        rank=3,
        model='nnmtf',
        nonnegative=True,  # Always True for MatrixTensorFactor
        metadata=metadata
    )
    print(f"   ✓ Factorization complete")
    print(f"   ✓ Result keys: {factorization_result.keys()}")
    print(f"   ✓ Final error: {factorization_result['error']:.6f}")
    print(f"   ✓ Converged: {factorization_result['converged']}")
    print(f"   ✓ Iterations: {factorization_result['iterations']}")

    # Step 6: Denormalize reconstruction
    print("\n6. Denormalizing reconstruction...")
    denormalized_reconstruction = tensor_factorization.denormalize_tensor(
        normalized_tensor=factorization_result['reconstruction'],
        normalization_params=norm_params
    )
    print(f"   ✓ Denormalized shape: {denormalized_reconstruction.shape}")

    # Step 7: Calculate R²
    print("\n7. Calculating explained variance...")
    r2 = tensor_factorization.explained_variance(tensor, denormalized_reconstruction)
    print(f"   ✓ R² = {r2:.4f}")

    print("\n" + "=" * 60)
    print("✓ FULL PIPELINE TEST PASSED!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
