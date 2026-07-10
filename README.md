# Multi-Label Chest X-Ray Pathology Classifier using Deep Learning

An end-to-end computer vision pipeline developed in PyTorch to classify 14 distinct thoracic pathologies from the NIH Chest X-ray dataset. The model addresses severe class imbalance utilizing custom threshold optimization, achieving a **0.7985 Mean AUC** on a single-batch subset evaluation.

## Key Highlights & Engineering Challenges Overcome
* **Secure Cloud Data Pipeline:** Integrated Kaggle API directly within a cloud environment, utilizing encrypted environment variables (`Colab Secrets`) to securely ingest high-volume medical data.
* **Advanced Fine-Tuning Strategy:** Employed a multi-phase training regimen—unfreezing deep convolutional layers with a highly granular learning rate—boosting the Mean AUC baseline from `0.7289` to `0.7896`.
* **Mitigating Class Imbalance:** Resolved the "Imbalanced Dataset Trap" where rare conditions (e.g., Hernia, Edema) yielded a `0.000` F1-score under standard `0.50` decision boundaries. Implemented a threshold grid-search maximizing Youden’s J-statistic to dynamically adjust decision cutoffs per pathology.

---

## Training Dynamics & Convergence
The network demonstrated clean convergence across 5 baseline epochs and 3 fine-tuning epochs without experiencing overfitting.

### Loss & AUC Trajectories

<img width="1389" height="490" alt="Unknown-2" src="https://github.com/user-attachments/assets/55af0dd5-6994-43e6-bf16-c81f6aa551b1" />

As visualized above, the training loss dipped cleanly beneath validation loss around Epoch 3, signaling a robust optimization shift without divergence.

---

## Performance Evaluation

### ROC-AUC per Pathology Breakdown
The model effectively distinguished structural anomalies with macro-geometric footprints (e.g., Cardiomegaly, Hernia) while identifying opportunities for future spatial optimization on localized micro-features (e.g., Nodules).

| Pathology | ROC-AUC | Rating | Pathology | ROC-AUC | Rating |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hernia** | 0.9880 | Excellent | **Effusion** | 0.8485 | Good |
| **Edema** | 0.9729 | Excellent | **Consolidation** | 0.8350 | Good |
| **Pneumothorax** | 0.8636 | Good | **Pleural Thickening** | 0.8130 | Good |
| **Cardiomegaly** | 0.8568 | Good | **Mass** | 0.7638 | Fair |

<img width="1189" height="590" alt="Model by condition" src="https://github.com/user-attachments/assets/06dd15ca-aa28-4eae-9be7-da951034a357" />


---

## Qualitative Analysis (Model Predictions on Test Scans)
To validate clinical reasoning, the model was evaluated on individual test films to map multi-label predictions against ground truth labels.

<img width="1143" height="1181" alt="Ind-Xrays" src="https://github.com/user-attachments/assets/07b9c435-fa88-47ad-9218-aa3911c0a42a" />

### Key Clinical Insights:
1. **Multi-Label Mastery:** In complex cases (e.g., *Effusion \| Infiltration*), the model simultaneously balanced co-occurring pathologies, accurately matching both clinical presentations.
2. **Artifact Sensitivity:** Qualitative analysis revealed classic deep learning shortcuts; the presence of intensive care monitoring hardware (ECG leads, portable text stamps) naturally elevated the model's baseline risk assessment for *Atelectasis*.
3. **Anatomical Overlaps:** In micro-features like *Nodules*, overlapping bone densities (clavicle/rib intersections) introduced minor false-positive signals, mimicking standard diagnostic challenges faced by early-career radiologists.

---

## Tech Stack & Tools
* **Language:** Python
* **Frameworks:** PyTorch, Torchvision
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Infrastructure:** Google Colab, Kaggle API CLI
