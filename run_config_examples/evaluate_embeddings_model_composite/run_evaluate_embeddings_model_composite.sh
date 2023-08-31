#!/bin/bash

# define the path to the script
script_path=$(dirname "$0")

# define the output directory
output_dir="$script_path/results"

# load configuration variables
source "$script_path/config/config.cfg"

train_data_file_p1=$(find "$script_path/data/" -name '*p1_train.txt' | head -n 1)
train_data_file_p2=$(find "$script_path/data/" -name '*p2_train.txt' | head -n 1)

test_data_file_p1=$(find "$script_path/data/" -name '*p1_test.txt' | head -n 1)
test_data_file_p2=$(find "$script_path/data/" -name '*p2_test.txt' | head -n 1)

param_grid_path="$script_path/config/param_grid.json"

if [[ -z "$train_data_file_p1" ]]; then
  echo "Error: no first segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$train_data_file_p2" ]]; then
  echo "Error: no second segment of train data file found in data directory."
  exit 1
fi

if [[ -z "$test_data_file_p1" ]]; then
  echo "Error: no first segment of test data file found in data directory."
  exit 1
fi

if [[ -z "$test_data_file_p2" ]]; then
  echo "Error: no second segment of test data file found in data directory."
  exit 1
fi

# get evaluation results
evaluate_command="python3 $script_path/../../classification_with_embeddings \
          evaluate-embeddings-model \
          --train-data-path $train_data_file_p1 $train_data_file_p2 \
          --test-data-path $test_data_file_p1 $test_data_file_p2 \
          --results-path $output_dir \
          --param-grid-path $param_grid_path \
          --no-grid-search "

if [[ -n "$method" ]]; then
  evaluate_command+="--method $method "

  if [[ "$method" == "starspace" ]]; then
    evaluate_command+="--starspace-path $script_path/../../embedding_methods/StarSpace/starspace "
  fi
fi

if [[ -n "$clf" ]]; then
  evaluate_command+="--internal-clf $clf "

  case "$clf" in
  "logistic-regression" | "random-forest" | "svc" | "gradient-boosting")
    if [[ -v internal_clf_args ]]; then
      evaluate_command+="--internal-clf-args \"${internal_clf_args//,/ }\""
    fi
    ;;
  *)
    false
    ;;
  esac
fi

echo "$evaluate_command"
eval "$evaluate_command"
