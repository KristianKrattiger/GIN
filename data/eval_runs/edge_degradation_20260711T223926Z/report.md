# Decode-in-the-loop degradation under a noisy edge

| scenario | edge | mode | fabrication | div_fidelity | supp_irrel | spurious |
|---|---|---|---|---|---|---|
| clean (agreeing pair, no edge) | none | convergent | 0.000 | n/a | 0.000 | False |
| NOISY class-C (agreeing pair, contradicts) | contradicts | divergent | 0.000 | 1.000 | 0.000 | True |
| control (real contradiction) | contradicts | divergent | 0.000 | 1.000 | 0.500 | False |

## Answers

- **clean (agreeing pair, no edge)** (convergent): ' In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, with acreage burned below both the five- and ten-year averages. |'
- **NOISY class-C (agreeing pair, contradicts)** (divergent): " In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, with acreage burned below both the five- and ten-year averages. | [1] About one-quarter of the nation's wildfires in 2023 occurred on federally protected lands. |"
- **control (real contradiction)** (divergent): ' In 2023, 56,580 wildfires burned 2,693,910 acres across the United States, with acreage burned below both the five- and ten-year averages. | [1] Elderly, immunocompromised, and low-income populations face heightened risk from wildfire smoke exposure. |'
