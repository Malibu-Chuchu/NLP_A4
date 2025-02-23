# A4 - Do You Agree???
## Patsachon Pattakulpong (st124952)

## 1. Load Data
- Preprocess text: remove special characters, convert to lowercase, and split into tokens.
- Dataset: [QQP Triplets](https://huggingface.co/datasets/embedding-data/QQP_triplets)
  - Contains 101,762 rows, suitable for training.
  - Structure:
    ```json
    {"query": [anchor], "pos": [positive], "neg": [negative1, negative2, ..., negativeN]}
    ```
  - Each example includes an anchor sentence, a positive sentence, and a list of negative sentences.
- To extract text data:
  ```python
  text = []
  for i in range(len(dataset['train'])):
      new_sets = [dataset['train'][i]['set']['query']] + dataset['train'][i]['set']['pos'] + dataset['train'][i]['set']['neg']
      text.extend(new_sets)
  ```
- The dataset is designed for training Sentence Transformers models.

### 2. Data Loader
- **Token Embedding**: Add `[CLS]` and `[SEP]` tokens.
- **Segment Embedding**: Differentiate sentence pairs.
- **Masking**: Randomly mask 15% of tokens.
- **Padding**: Pad sequences to `max_len`.

### 3. Model
BERT consists of:
- Embedding layers (token, position, segment)
- Multi-head attention & attention mask
- Position-wise feed-forward network
- Encoder layers

### 4. Training
- Uses masked language modeling (MLM) & next sentence prediction (NSP).

### 5. Save model and get bert_model.pth

## 2. Evaluation and Analysis
### Performance Table
| Model Type          | SNLI and MNLI Performance (Accuracy)|
|---------------------|-----------|
| S-BERT Model          | 30.7%   |
## S-BERT Model for SNLI + MNLI Classification
Implementation of the S-BERT model, trained on the **SNLI** (Stanford Natural Language Inference) and **MNLI** (Multi-Genre Natural Language Inference) datasets for sentence pair classification. The goal is to predict the relationship (entailment, contradiction, or neutral) between two sentences: a premise and a hypothesis.
## Model Performance
The S-BERT model, trained on SNLI + MNLI datasets, achieved the following performance:
- **Accuracy:** 30.7% on the combined task of SNLI and MNLI classification.
## Training Process
The training process is as follows:
1. **Model:** use the model from BERT embeddings for the premise and hypothesis sentences which is bert_model.pth, followed by mean pooling to extract the relevant features. The features are then passed through a classifier head for the final prediction.
2. **Epochs:** The model is trained for **5 epochs** with a **batch size of 8**.
3. **Data:** The training data consists of premise-hypothesis pairs labeled as one of the following:
    - **Entailment**
    - **Contradiction**
    - **Neutral**
4. **Loss Function:** The model uses **cross-entropy loss** to calculate the difference between predicted and true labels.
5. **Optimization:** The model parameters are updated using **backpropagation**, with **Adam optimizer** and **learning rate scheduling**.
6. **Accuracy:** The accuracy is computed after each epoch by comparing the model's predictions with the true labels.

## Challenges and Limitations
- The model achieved only **30.7% accuracy**, indicating potential issues with underfitting, model complexity, or optimization. This result suggests that the model misclassified instances, incorrectly labeling them as **Entailment** instead of **Contradiction**.
- **Limited Batch Size:** A batch size of **8** might hinder convergence and generalization.
- **Simplified Architecture:** The model uses **mean pooling** for feature extraction, which may not fully capture the premise-hypothesis relationship.
- **Training Time:** The model was trained for **5 epochs**, which may be insufficient for complex NLI tasks.

## Proposed Improvements
1. **Increase Training Time:** Train for more epochs (e.g., 10–20) and experiment with a larger **batch size** (16–32).
2. **Model Enhancements:**
   - Use **attention mechanisms** (e.g., self-attention, cross-attention) or more advanced models.
   - Consider another architecture to make a better model sentence for pair relationships.

## Application Demo
- You can run: run **python app.py** !! but you have to download and move models in GG drive link to the model file !!

## VDO 
- You can watch application demo via this link: https://www.youtube.com/watch?v=rzzxpuQ9dGk
- FYI: Since .pth is too large you can access all my model via this drive link: https://drive.google.com/drive/folders/18tTuJUr7iRipc1gAhtwPqAqnkuRyYab9?usp=sharing

  









