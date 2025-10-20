# SMS (Simultaneous Multi-Slice) Support in SVR Pipeline

## Overview

This implementation adds support for Simultaneous Multi-Slice (SMS/Multiband) MRI acquisitions to the SVR reconstruction pipeline. In SMS acquisitions, multiple slices are excited and acquired simultaneously, which means they share the same motion state and should be registered together as a group.

## Key Features

### 1. SMS Metadata Storage
- **JSON Sidecars**: Stack metadata is stored as `.json` files alongside NIfTI files
- **Metadata Fields**:
  - `mb_factor`: Multiband factor (1 = no SMS, >1 = number of simultaneous slices)
  - `acquisition_order`: Slice acquisition order ('interleaved-odd-even', 'sequential-asc', etc.)
  - Additional simulation parameters (noise, displacement, motion jitter)

### 2. Stack Class Extensions
The `Stack` class in `standalone_inlined/image.py` now includes:
- `mb_factor`: Multiband acceleration factor
- `acquisition_order`: Acquisition sequence pattern

These attributes are:
- Preserved through clone/copy operations
- Passed through substack extraction
- Automatically loaded from JSON sidecars when loading NIfTI stacks

### 3. SMS-Aware Registration
The `SliceToVolumeRegistration` class in `standalone_inlined/svr/registration.py` now:
- Detects SMS stacks via the `mb_factor` attribute
- Groups slices based on SMS acquisition pattern
- Applies shared rigid transformations to simultaneously acquired slices
- Averages registration parameters within each SMS group

#### SMS Grouping Logic
```python
def _build_sms_groups(nz: int, mb_factor: int, acq_order: Optional[str]) -> List[List[int]]
```
- Builds groups of slices acquired simultaneously
- Respects acquisition order (interleaved/sequential)
- Spaces slices across the slab using modulo-based grouping
- For mb_factor=3 and nz=12 with interleaved-odd-even:
  - Group 0: slices [0, 6, 3, 9]
  - Group 1: slices [4, 10, 1, 7]
  - Group 2: slices [2, 8, 5, 11]

### 4. Simulation Support
The `simulate_stacks_from_mri.py` script now:
- Generates SMS stacks with proper grouping
- Applies shared motion within each SMS group
- Saves metadata as JSON sidecars
- Command-line flags:
  - `--mb-factor`: SMS acceleration factor (default: 1)
  - `--acq-order`: Acquisition order (default: interleaved-odd-even)
  - `--max-rot-deg`: Per-slice rotation jitter
  - `--max-trans-mm`: Per-slice translation jitter

## Usage

### Generating SMS Stacks for Testing

```bash
python simulate_stacks_from_mri.py \
    input_volume.nii.gz \
    output_dir \
    --n-stacks 5 \
    --slices-per-stack 12 \
    --mb-factor 3 \
    --acq-order interleaved-odd-even \
    --max-rot-deg 2 \
    --max-trans-mm 0.5 \
    --noise-std 0.01 \
    --max-disp 1
```

### Running SVR with SMS Stacks

```bash
python svr_cli.py \
    output_dir/sim_stack_*.nii.gz \
    -o svr_output.nii.gz \
    --device 0
```

The pipeline will automatically:
1. Load SMS metadata from JSON sidecars
2. Apply SMS constraints during slice-to-volume registration
3. Ensure simultaneously acquired slices maintain shared transformations

### Verifying SMS Metadata

```bash
python test_sms_loading.py
```

This will:
- Load an SMS stack
- Display SMS metadata
- Show the computed SMS groups
- Verify correct metadata loading

## Implementation Details

### Modified Files

1. **standalone_inlined/image.py**
   - Added `mb_factor` and `acquisition_order` to `Stack.__init__`
   - Updated `_clone_dict`, `like`, and `get_substack` methods
   - Modified `load_stack` to read JSON sidecars

2. **svr_cli.py**
   - Updated `load_stack` to read JSON sidecars
   - Maintains compatibility with non-SMS stacks

3. **standalone_inlined/svr/registration.py**
   - Added `_build_sms_groups` helper function
   - Modified `SliceToVolumeRegistration.forward` to apply SMS constraints
   - Averages transformations within SMS groups post-registration

4. **simulate_stacks_from_mri.py**
   - Added JSON sidecar export
   - Includes all simulation parameters in metadata

### SMS Constraint Application

During registration:
1. Individual slices are registered independently to the volume
2. After registration, transformations are grouped by SMS pattern
3. Within each group, rigid transformation parameters are averaged
4. The averaged transformation is applied to all slices in the group
5. This ensures physically consistent motion for simultaneously acquired slices

### Backward Compatibility

- Stacks without JSON sidecars default to `mb_factor=1` (no SMS)
- Non-SMS stacks are processed identically to before
- The SMS constraint code only activates when `mb_factor > 1`

## Testing

### Test Script
`test_sms_loading.py` verifies:
- JSON sidecar loading
- SMS group construction
- Metadata propagation through Stack operations

### Expected Output
```
Stack properties:
  Shape: torch.Size([12, 1, 148, 174])
  Resolution: 0.800 x 0.800 x 8.200 mm
  MB Factor: 3
  Acquisition Order: interleaved-odd-even

SMS groups (mb_factor=3):
  Group 0: slices [0, 6, 3, 9]
  Group 1: slices [4, 10, 1, 7]
  Group 2: slices [2, 8, 5, 11]
```

## Future Enhancements

Potential improvements:
1. **CAIPI Patterns**: Support controlled aliasing patterns
2. **Custom Slice Tables**: Vendor-specific acquisition sequences
3. **Metadata Validation**: Check for physically plausible SMS parameters
4. **Extended NIfTI Headers**: Store metadata in NIfTI extensions (alternative to JSON)
5. **Multi-Echo SMS**: Handle multi-echo SMS acquisitions
6. **Blipped-CAIPI**: Simulate gradient blips between SMS slices

## References

- Barth M, et al. (2016). "Simultaneous multislice (SMS) imaging techniques." Magnetic Resonance in Medicine.
- Setsompop K, et al. (2012). "Blipped-controlled aliasing in parallel imaging for simultaneous multislice echo planar imaging with reduced g-factor penalty." Magnetic Resonance in Medicine.
