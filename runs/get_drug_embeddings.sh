#!/bin/bash

mkdir -p embeddings/pca
mkdir -p embeddings/umap
mkdir -p embeddings/autoencoder

# input file placeholder
INPUT_FILE="gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/lola/EMR_HW_11272024.parquet"
OUTPUT_DIR="gs://fc-secure-19ab668e-266f-4a5f-9c63-febea17b23cf/lola/drug_embeddings"
CACHE_DIR=".cache"

# run embeddings generation for different dimensions and reduction methods
# pca - 8 dimensions
python get_drug_embeddings.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/pca/drug_embeddings_8dim.parquet \
    --model_name UFNLP/gatortron-base \
    --embedding_dim 8 \
    --batch_size 32 \
    --reduction_method pca \
    --cache_file ${CACHE_DIR}/gatortron_base_embed_cache.parquet

# pca - 32 dimensions
python get_drug_embeddings.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/pca/drug_embeddings_32dim.parquet \
    --model_name UFNLP/gatortron-base \
    --embedding_dim 32 \
    --batch_size 32 \
    --reduction_method pca \
    --cache_file ${CACHE_DIR}/gatortron_base_embed_cache.parquet

# umap - 8 dimensions
python get_drug_embeddings.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/umap/drug_embeddings_8dim.parquet \
    --model_name UFNLP/gatortron-base \
    --embedding_dim 8 \
    --batch_size 32 \
    --reduction_method umap \
    --cache_file ${CACHE_DIR}/gatortron_base_embed_cache.parquet

# umap - 32 dimensions
python get_drug_embeddings.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/umap/drug_embeddings_32dim.parquet \
    --model_name UFNLP/gatortron-base \
    --embedding_dim 32 \
    --batch_size 32 \
    --reduction_method umap \
    --cache_file ${CACHE_DIR}/gatortron_base_embed_cache.parquet

# autoencoder - 8 dimensions
python get_drug_embeddings.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/autoencoder/drug_embeddings_8dim.parquet \
    --model_name UFNLP/gatortron-base \
    --embedding_dim 8 \
    --batch_size 32 \
    --reduction_method autoencoder \
    --cache_file ${CACHE_DIR}/gatortron_base_embed_cache.parquet

# autoencoder - 32 dimensions
python get_drug_embeddings.py \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_DIR}/autoencoder/drug_embeddings_32dim.parquet \
    --model_name UFNLP/gatortron-base \
    --embedding_dim 32 \
    --batch_size 32 \
    --reduction_method autoencoder \
    --cache_file ${CACHE_DIR}/gatortron_base_embed_cache.parquet
