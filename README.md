# Unofficial W-CFM (Weighted Conditional Flow Matching) Implementation

This is an **unofficial implementation** of Weighted Conditional Flow Matching (W-CFM) based on the paper:

**"Weighted Conditional Flow Matching"**  
*Sergio Calvo-Ordoñez, Matthieu Meunier, Álvaro Cartea, Christoph Reisinger, Yarin Gal, Jose Miguel Hernández-Lobato*  
arXiv: [2507.22270](https://arxiv.org/abs/2507.22270)

## 📚 Purpose

This implementation is created for **educational and research purposes only**. It aims to:

- Provide a practical implementation of the W-CFM algorithm for learning and experimentation
- Help researchers understand the core concepts of weighted flow matching
- Serve as a starting point for further research in flow-based generative models
- Demonstrate the simplicity of incorporating entropic optimal transport insights into CFM

## 🔬 What is W-CFM?

W-CFM is a novel approach to training continuous normalizing flows that improves upon standard Conditional Flow Matching (CFM) by incorporating a simple Gibbs kernel weight:

```python
w_ε(x, y) = exp(-||x - y|| / ε)
```

This weighting scheme:

- ✨ Approximates entropic optimal transport without expensive computations
- 🚀 Produces straighter flow paths leading to faster and more accurate generation
- 💪 Maintains computational efficiency of vanilla CFM
- 📊 Shows improved performance on multimodal datasets

⚠️ Disclaimer
This is an unofficial implementation and may differ from the authors' original code. The implementation is based on our interpretation of the paper and is provided "as-is" for research purposes.
🎯 Key Features

Simple modification to standard CFM training
No additional computational overhead
Improved sample quality and generation efficiency
Better handling of multimodal distributions

🚀 Quick Start
```python
# Key modification in the training loop
def compute_gibbs_weight(x0, x1, epsilon):
    """Compute W-CFM weights"""
    distances = torch.norm(x1.flatten(1) - x0.flatten(1), dim=1)
    return torch.exp(-distances / epsilon)

# Apply weights to loss
gibbs_weights = compute_gibbs_weight(noise, data, epsilon=5.0)
weighted_loss = (gibbs_weights * per_sample_loss).mean()
```

🔧 Implementation Details

- Epsilon Selection: For CIFAR-10, we use ε = 5.0 as reported in the paper
- High-dimensional Heuristic: Can also use ε = κ√d where d is the data dimension
- Weight Monitoring: Track Gibbs weight statistics to ensure proper behavior

📖 Citation
If you find this implementation helpful for your research, please cite the original paper:
```
bibtex@article{calvo2025weighted,
  title={Weighted Conditional Flow Matching},
  author={Calvo-Ordo{\~n}ez, Sergio and Meunier, Matthieu and Cartea, {\'A}lvaro and 
          Reisinger, Christoph and Gal, Yarin and Hern{\'a}ndez-Lobato, Jose Miguel},
  journal={arXiv preprint arXiv:2507.22270},
  year={2025}
}
```
🤝 Contributing

This implementation is for learning purposes. Feel free to open issues for discussions about the method or potential improvements. For official implementations or clarifications about the method, please refer to the original authors.

📝 License

This code is provided for educational and research purposes. Please check with the original authors regarding any specific usage restrictions.