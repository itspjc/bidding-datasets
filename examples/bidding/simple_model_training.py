#!/usr/bin/env python
# coding: utf-8

# # Simple Model Training Example
#
# This example is the API example for this Ludwig command line example
# (https://ludwig-ai.github.io/ludwig-docs/examples/#kaggles-titanic-predicting-survivors).

# Import required libraries

from ludwig.api import LudwigModel
import logging
import shutil


# clean out prior results
try:
    shutil.rmtree('./results')
except FileNotFoundError:
    pass


# Define Ludwig model object that drive model training
model = LudwigModel(model_definition_file='./mymodel_definition.yaml',
                    logging_level=logging.INFO)

# initiate model training
train_stats = model.train(data_csv='./data/train_0605.csv',
                          experiment_name='simple_experiment',
                          model_name='simple_model')

model.close()