# Ownership Verification in Deep Neural Networks via Watermarking

## 📌 Overview

As deep neural networks (DNNs) become central to a wide range of real-world applications, the need to protect these valuable models from unauthorized use and intellectual property (IP) theft is increasingly critical. This project investigates watermarking techniques as a means of ownership verification for convolutional neural networks (CNNs), focusing on preserving model functionality while offering reliable IP protection.

## 🎯 Objective

The primary goal of this project is to evaluate and compare different watermarking methods for CNNs with respect to:
- Robustness under adversarial attacks (e.g., pruning, knowledge distillation)
- Stealthiness and undetectability
- Functional impact and performance trade-offs

## 🧪 Techniques Explored

We implemented and analyzed the following watermarking approaches:

1. **Backdoor Trigger-Based Watermarking**
   - Embeds unique input-output behavior using trigger patterns.
   - Robust under pruning; fails under knowledge distillation.

2. **Weight Perturbation**
   - Introduces small, structured modifications to model weights.
   - Maintains moderate robustness under pruning; fails under distillation.

3. **Passport-Based Watermarking**
   - Requires valid cryptographic credentials (passports) at inference time.
   - Uniquely supports intentional performance degradation when used without a passport.
   - Adds an additional layer of deterrence and control.

## 🧩 Methodology

Our study was conducted in two main phases:

1. **Baseline Model Development**
   - A CNN model was trained and evaluated on a standard classification task.

2. **Watermark Embedding & Evaluation**
   - Each watermarking method was applied to the baseline model.
   - Models were subjected to attacks including weight pruning and knowledge distillation.
   - Metrics such as F1 score and watermark verification accuracy were used for evaluation.

## 📊 Results Summary

| Method               | Pruning Robustness | Distillation Robustness | Unique Features                             |
|----------------------|--------------------|--------------------------|---------------------------------------------|
| Backdoor Trigger     | ✅ High             | ❌ Low                   | 100% verification accuracy post-pruning     |
| Weight Perturbation  | ✅ High        | ❌ Low                   | ~70% verification accuracy post-pruning     |
| Passport-Based       | ✅ High         | ❌ Moderate              | Controlled performance degradation on fail  |

These results highlight important trade-offs between security, robustness, and usability when selecting a watermarking method for deployment.

## 📁 Project Structure

```
├── models/                # CNN model implementations
├── watermarking/          # Watermark embedding and verification code
├── evaluation/            # Scripts for pruning, distillation, and analysis
├── results/               # Experimental results and plots
├── utils/                 # Helper functions
├── README.md              # Project description
└── requirements.txt       # Python dependencies
```

## 📌 Key Takeaways

- **No one-size-fits-all**: Each method has strengths and weaknesses depending on the threat model and deployment scenario.
- **Passport-based watermarking** introduces a novel control mechanism that deters unauthorized usage via intentional model degradation.
- Robust watermarking requires balancing stealthiness, reliability, and performance impact.

## 📚 References

- [1] Lederer et al., IEEE Trans. Neural Netw. Learn. Syst., 2023
- [2] Soremekun et al., Computers & Security, 127, 2023. https://doi.org/10.1016/j.cose.2023.103101
- [3] Fan et al., IEEE TPAMI, 44(10), 2022. https://doi.org/10.1109/TPAMI.2021.3088846
