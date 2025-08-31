# Known Issues

## Flash Attention Compatibility with GLUE Evaluation

### Issue
Flash attention (via xformers' `memory_efficient_attention`) is not compatible with GLUE evaluation tasks due to memory alignment requirements.

### Technical Details
- xformers requires sequence lengths to be aligned to multiples of 8 for optimal memory access
- GLUE tasks use variable-length sequences with dynamic batching
- Padding sequences to multiples of 8 would require complex mask adjustments and output trimming
- The alignment issue causes incorrect attention computation leading to ~50% accuracy (random performance)

### Symptoms
When flash attention is enabled for GLUE evaluation:
- Model performance drops to random chance (~50% accuracy for binary tasks)
- Matthews correlation becomes 0.0 for CoLA
- All tasks show severely degraded performance

### Workaround
The GLUE trainer (`src/neobert/glue/train.py`) automatically disables flash attention and uses eager attention instead. If flash attention is specified in the configuration, a warning will be logged:
```
Flash attention is not supported for GLUE evaluation due to memory alignment issues with variable-length sequences. Using eager attention instead.
```

### Impact
- GLUE evaluation runs slightly slower without flash attention optimization
- No impact on model accuracy - eager attention produces correct results
- Pretraining and other tasks can still use flash attention normally

### Future Resolution
A proper fix would require:
1. Dynamic padding of sequences to multiples of 8 during forward pass
2. Proper mask adjustment for padded positions
3. Trimming output tensors back to original sequence lengths
4. Extensive testing to ensure correctness

For now, the automatic fallback to eager attention ensures correct GLUE evaluation results.