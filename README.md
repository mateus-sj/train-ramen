# train-ramen
Repositório para a realização do alinhamento de modelos tipo roberta utilizando o ramen - Português jurídico.

Baseado no artigo [From English To Foreign Languages: Transferring Pre-trained Language Models](https://arxiv.org/abs/2002.07306).

# Docker Build
```
docker build -t train_roberta_ramen .
```
# Docker Run
```
docker run -v /path/to/host/binary_data:/work/binary_data \
           -v /path/to/host/roberta_pretrained:/work/models/roberta_pretrained \
           -v /path/to/host/ramen_jur_roberta:/work/models/ramen_jur_roberta \
           train_roberta_ramen
```