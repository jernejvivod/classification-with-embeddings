#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config_get_entity_embeddings_composite.cfg"

train_data_file_p1=$(find "$script_path/data/" -name '*p1_train.txt' | head -n 1)
train_data_file_p2=$(find "$script_path/data/" -name '*p2_train.txt' | head -n 1)

if [[ -z "$train_data_file_p1" ]]; then
  echo "Error: no first segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$train_data_file_p2" ]]; then
  echo "Error: no second segment of train data file found in data directory."
  exit 1
fi

# get embeddings
get_embeddings_command_1="python3 $script_path/../../classification_with_embeddings \
          get-entity-embeddings \
          --train-data-path $train_data_file_p1 \
          --output-dir $output_dir "

get_embeddings_command_2="python3 $script_path/../../classification_with_embeddings \
          get-entity-embeddings \
          --train-data-path $train_data_file_p2 \
          --output-dir $output_dir "

if [[ -n "$method" ]]; then
  get_embeddings_command_1+="--method $method "
  get_embeddings_command_2+="--method $method "

  if [[ "$method" == "word2vec" && -n "$word2vec_args" ]]; then
    get_embeddings_command_1+="--word2vec-args \"${word2vec_args//,/ }\" "
    get_embeddings_command_2+="--word2vec-args \"${word2vec_args//,/ }\" "
  fi

  if [[ "$method" == "fasttext" && -n "$fasttext_args" ]]; then
    get_embeddings_command_1+="--fasttext-args  \"${fasttext_args//,/ }\""
    get_embeddings_command_2+="--fasttext-args  \"${fasttext_args//,/ }\""
  fi

  if [[ "$method" == "starspace" && -n "$starspace_args" ]]; then
    get_embeddings_command_1+="--starspace-args \"${starspace_args//,/ }\" "
    get_embeddings_command_2+="--starspace-args \"${starspace_args//,/ }\" "
  fi
fi

echo "$get_embeddings_command_1"
eval "$get_embeddings_command_1"
mv "$output_dir"/*model.tsv "$output_dir/embeddings_p1.tsv"

echo "$get_embeddings_command_2"
eval "$get_embeddings_command_2"
mv "$output_dir"/*model.tsv "$output_dir/embeddings_p2.tsv"
