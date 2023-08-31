#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config_train_cnn_model_composite.cfg"

train_data_file_p1=$(find "$script_path/data/" -name '*p1_train.txt' | head -n 1)
train_data_file_p2=$(find "$script_path/data/" -name '*p2_train.txt' | head -n 1)

val_data_file_p1=$(find "$script_path/data/" -name '*p1_val.txt' | head -n 1)
val_data_file_p2=$(find "$script_path/data/" -name '*p2_val.txt' | head -n 1)

word_embeddings_file_p1=$(find "$script_path/results/" -name '*p1.tsv' | head -n 1)
word_embeddings_file_p2=$(find "$script_path/results/" -name '*p2.tsv' | head -n 1)

if [[ -z "$train_data_file_p1" ]]; then
  echo "Error: no first segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$train_data_file_p2" ]]; then
  echo "Error: no second segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$word_embeddings_file_p1" ]]; then
  echo "Error: no stored word embeddings for the first segment found in the results directory."
  exit 1
fi

if [[ -z "$word_embeddings_file_p2" ]]; then
  echo "Error: no stored word embeddings for the second segment found in the results directory."
  exit 1
fi

if [[ -z "$n_labels" ]]; then
  echo "Error: the number of unique labels in the dataset should be specified in config file."
  exit 1
fi

# get evaluation results
train_command="python3 $script_path/../../classification_with_embeddings \
          train-cnn-model \
          --train-data-path $train_data_file_p1 $train_data_file_p2 \
          --word-embeddings-path $word_embeddings_file_p1 $word_embeddings_file_p2 \
          --n-labels $n_labels \
          --output-dir $output_dir "

if [[ -f "$val_data_file_p1" ]]; then
  train_command+="--val-data-path $val_data_file_p1 $val_data_file_p2"
fi

if [[ -n "$batch_size" ]]; then
  train_command+="--batch-size $batch_size "
fi

if [[ -n "$n_epochs" ]]; then
  train_command+="--n-epochs $n_epochs "
fi

if [[ -n "$max_filter_s" ]]; then
  train_command+="--max-filter-s $max_filter_s "
fi

if [[ -n "$min_filter_s" ]]; then
  train_command+="--min-filter-s $min_filter_s "
fi

if [[ -n "$filter_s_step" ]]; then
  train_command+="--filter-s-step $filter_s_step "
fi

if [[ -n "$n_filter_channels" ]]; then
  train_command+="--n-filter-channels $n_filter_channels "
fi

if [[ -n "$hidden_size" ]]; then
  train_command+="--hidden-size $hidden_size "
fi

echo "$train_command"
eval "$train_command"
